"""
Defines WorkerPool for managing multiple WorkerProcessManagers
"""

import asyncio
import signal
import time
from dataclasses import dataclass
from itertools import product
from typing import Any, TypeVar, cast

from src.shared.logger import Logger
from src.shared.utils.worker_pool.worker_process_manager import (
    ROLLING_UTILIZATION_WINDOW_NS,
    JobHandle,
    WorkerProcessManager,
    WorkerSnapshot,
)

from .job_context_interface import JobContextInterface
from .job_interface import JobInterface
from .job_result import JobObserver

C = TypeVar("C", bound=tuple)
D = TypeVar("D")
R = TypeVar("R")
Conf = TypeVar("Conf")

#: How often the supervisor checks every worker process is still alive
WORKER_SUPERVISION_INTERVAL_SEC = 1.0
#: Wait between attempts to build a replacement worker after one failed
WORKER_RESPAWN_RETRY_SEC = 5.0
#: Exit codes of a worker that was asked to stop rather than one that died:
#: a clean exit, or termination by the signals a shutdown delivers to the
#: whole process group (SIGTERM, SIGINT, SIGHUP). Such a worker is not
#: replaced: the service itself is going down and the pool's own shutdown
#: follows.
ORDERLY_EXIT_CODES = frozenset(
    {0, -signal.SIGTERM, -signal.SIGINT, -signal.SIGHUP}
)


@dataclass
class ContextAssignment:
    """
    Assigns a context definition to a specific set of workers
    Each listed worker will eagerly create an instance of this context at startup.
    """

    context_def: JobContextInterface[Any]
    worker_ids: list[int]


class WorkerPool:
    """
    Interface for managing multiple WorkerProcessManagers
    Assigns contexts to workers up front and routes jobs to workers that own
    a matching context with lowest utilization.

    Usage
    ```
    class Context(JobContextInterface[int]):
        def __init__(self):
            super().__init__(tags=["some_context"])

        def create(self, log: Logger) -> int:
            return 42

        def destroy(self, log: Logger, context: int) -> None:
            return


    class Job(JobInterface[tuple[int], int, int]):
        def process_batch(
            self, log: Logger, contexts: tuple[int], batch: list[int]
        ) -> int:
            return sum(batch) + contexts[0]


    pool = WorkerPool(
        logger,
        num_workers=2,
        contexts=[ContextAssignment(Context(), worker_ids=[0, 1])],
    )

    handle = pool.register_job(("some_context",), period_ms=100, job=Job())
    handle.queue_data([1, 2, 3])
    handle.on(handle.JobResultEvent, lambda result: print(result))

    # ...

    handle.deregister()
    pool.shutdown()
    ```
    """

    def __init__(
        self,
        logger: Logger,
        num_workers: int,
        contexts: list[ContextAssignment],
        rolling_utilization_window_ns: int = ROLLING_UTILIZATION_WINDOW_NS,
        job_observer: JobObserver | None = None,
        supervise: bool = True,
        supervision_interval_sec: float = WORKER_SUPERVISION_INTERVAL_SEC,
    ):
        """
        Args:
            logger      - Application logger
            num_workers - Number of worker processes to spawn
            contexts    - Context assignments. Each context_def is created on
                            every worker listed in its worker_ids. Position in
                            the list determines the context_id.
            rolling_utilization_window_ns - Override for the per-worker utilization
                                              smoothing window (production should
                                              use the default)
            job_observer - Optional callback invoked for every completed job
                            execution on every worker
            supervise   - Watch every worker process and replace one that
                            dies (see `_supervise`). Needs a running event
                            loop; without one nothing is watched
            supervision_interval_sec - How often liveness is checked

        Raises:
            ValueError      if num_workers is 0 or worker_ids reference an invalid worker
            RuntimeError    if any worker fails to initialize a context
        """
        if num_workers <= 0:
            raise ValueError("num_workers must be at least 1")

        self._log = logger
        self._rolling_utilization_window_ns = rolling_utilization_window_ns
        self._job_observer = job_observer
        self._per_worker_defs: list[dict[int, JobContextInterface[Any]]] = []
        # Replacements made per worker id, and the last one's wall time
        self.worker_restarts: dict[int, int] = {}
        self.last_worker_restart_at: float | None = None
        self._supervisor: asyncio.Task | None = None
        self._respawning: set[int] = set()
        self._shutting_down = False

        # Assign each context to its target workers (position in `contexts` is context_id)
        per_worker_defs: list[dict[int, JobContextInterface[Any]]] = [
            {} for _ in range(num_workers)
        ]
        for context_id, assignment in enumerate(contexts):
            for worker_id in assignment.worker_ids:
                if not 0 <= worker_id < num_workers:
                    raise ValueError(
                        f"Context {context_id} references invalid worker_id={worker_id}"
                    )
                per_worker_defs[worker_id][context_id] = assignment.context_def

        self._contexts = contexts
        self._per_worker_defs = per_worker_defs

        # Built by appending rather than as a list comprehension, and unwound on
        # failure, because each WorkerProcessManager spawns an OS process in its
        # own constructor and any of them can raise - a worker whose context
        # fails to load exits during initialization and the manager turns that
        # into a RuntimeError.
        #
        # A comprehension would discard the whole list on that raise, so the
        # workers already spawned would be left running with nothing holding a
        # reference to them: not reachable through self._processes (never
        # assigned), and not reachable by the caller (no object was returned).
        # They then outlive the process that made them and, being non-daemon,
        # stop the interpreter from exiting - which is how a missing model
        # dependency shows up as a test suite that completes every test and then
        # hangs forever with dozens of orphans.
        self._processes: list[WorkerProcessManager] = []
        try:
            for worker_id in range(num_workers):
                self._processes.append(
                    WorkerProcessManager(
                        logger,
                        worker_id,
                        per_worker_defs[worker_id],
                        rolling_utilization_window_ns,
                        job_observer,
                    )
                )
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt part-way through
            # spawning leaks processes exactly as an ImportError does, and the
            # bare `raise` means nothing is swallowed either way.
            self.shutdown()
            raise

        # Pre-compute tag -> set of context_ids that have it for fast routing
        self._tag_to_context_ids: dict[str, set[int]] = {}
        for context_id, assignment in enumerate(contexts):
            for tag in assignment.context_def.tags:
                self._tag_to_context_ids.setdefault(tag, set()).add(context_id)

        if supervise:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self._supervisor = loop.create_task(
                    self._supervise(supervision_interval_sec),
                    name="worker-pool-supervisor",
                )

    @property
    def num_workers(self) -> int:
        """
        Gets number of worker processes in the pool
        """
        return len(self._processes)

    def context_defs_for_tag(self, tag: str) -> list[JobContextInterface[Any]]:
        """
        The context definitions configured under a tag, so a provider can
        check at start-up that a tag it was given points at the kind of
        context its job needs

        Args:
            tag     - Context tag

        Returns:
            The definitions, in context-id order; empty for an unknown tag
        """
        return [
            self._contexts[context_id].context_def
            for context_id in sorted(self._tag_to_context_ids.get(tag, set()))
        ]

    def context_runtime_info(self, context_id: int) -> dict[str, Any]:
        """
        What a context reported about itself once created on a worker
        (`JobContextInterface.runtime_info`), from the first live worker that
        owns it

        Args:
            context_id  - Position of the context in the assignments

        Returns:
            The info, or an empty dict when no live worker owns the context
            or it reported nothing
        """
        for process in self._processes:
            if context_id in process.context_ids and process.alive:
                info = process.context_info.get(context_id)
                if info:
                    return dict(info)
        return {}

    def context_runtime_info_by_tag(self, tag: str) -> dict[str, Any]:
        """
        `context_runtime_info` for the first context under a tag that
        reported anything
        """
        for context_id in sorted(self._tag_to_context_ids.get(tag, set())):
            info = self.context_runtime_info(context_id)
            if info:
                return info
        return {}

    async def _supervise(self, interval_sec: float) -> None:
        """
        Watches every worker process and replaces one that died.

        A worker that dies after initialization used to be invisible: its
        jobs never completed and never raised, readiness went 503 and stayed
        there, and `_assign_process` kept routing new sessions to it. Now,
        once a second, a dead worker is abandoned (every job registered to
        it gets `JobHandle.JobLostEvent`, so its owner can register a new one),
        and a replacement with the same contexts is built on an executor
        thread, since loading a model blocks for seconds and the event loop
        is serving live sessions. Until the replacement is up, `alive` is
        False for that worker id and routing skips it; a job owner that
        retries `register_job` succeeds the moment it is back. The caption
        job of a session on a dead worker is not re-registered by the
        provider today (upstream's session has no such path); the diarization
        job is.
        """
        try:
            while not self._shutting_down:
                await asyncio.sleep(interval_sec)
                for worker_id, process in enumerate(self._processes):
                    if process.alive or worker_id in self._respawning:
                        continue
                    if process.exitcode in ORDERLY_EXIT_CODES:
                        # Asked to stop (the service is shutting down and the
                        # signal reached the worker first): not a death, and
                        # a replacement would only have to be torn down again.
                        continue
                    self._respawning.add(worker_id)
                    asyncio.get_running_loop().create_task(
                        self._replace_worker(worker_id, process),
                        name=f"worker-pool-respawn-{worker_id}",
                    )
        except asyncio.CancelledError:
            return

    async def _replace_worker(
        self, worker_id: int, dead: WorkerProcessManager
    ) -> None:
        """
        Abandons a dead worker's manager and builds its replacement, retrying
        every WORKER_RESPAWN_RETRY_SEC while the replacement fails to come up
        (a model that no longer loads): the pool keeps serving the other
        workers meanwhile and readiness keeps reporting the dead worker
        """
        try:
            lost = dead.abandon()
            self._warn(
                f"Worker {worker_id} exited unexpectedly (exit code "
                f"{dead.exitcode}); {lost} registered job(s) lost, spawning "
                "a replacement with the same contexts"
            )
            loop = asyncio.get_running_loop()
            while not self._shutting_down:
                started = time.perf_counter()
                try:
                    replacement = await loop.run_in_executor(
                        None,
                        lambda: WorkerProcessManager(
                            self._log,
                            worker_id,
                            self._per_worker_defs[worker_id],
                            self._rolling_utilization_window_ns,
                            self._job_observer,
                            loop,
                        ),
                    )
                # pylint: disable-next=broad-exception-caught
                except Exception as error:
                    self._warn(
                        f"Replacement for worker {worker_id} failed to start: "
                        f"{error}; retrying in {WORKER_RESPAWN_RETRY_SEC:.0f}s"
                    )
                    await asyncio.sleep(WORKER_RESPAWN_RETRY_SEC)
                    continue
                if self._shutting_down:
                    replacement.send_terminate()
                    replacement.wait_shutdown()
                    return
                self._processes[worker_id] = replacement
                self.worker_restarts[worker_id] = (
                    self.worker_restarts.get(worker_id, 0) + 1
                )
                self.last_worker_restart_at = time.time()
                self._info(
                    f"Worker {worker_id} replaced in "
                    f"{time.perf_counter() - started:.1f}s "
                    f"(restart {self.worker_restarts[worker_id]} of this "
                    "worker id)"
                )
                return
        finally:
            self._respawning.discard(worker_id)

    def _warn(self, message: str) -> None:
        if self._log is not None:
            self._log.warning(message)

    def _info(self, message: str) -> None:
        if self._log is not None:
            self._log.info(message)

    def worker_snapshots(self) -> list[WorkerSnapshot]:
        """
        Gets a point-in-time view of every worker's load, in worker id order

        Side effect free, so it is safe to call from a request handler.
        """
        return [process.snapshot() for process in self._processes]

    def load_for_tags(
        self, context_tags: tuple[str, ...]
    ) -> list[WorkerSnapshot]:
        """
        Gets the live workers that own EVERY given context tag, with their load

        The read-only counterpart to `_assign_process`, which answers the same
        routing question but *raises* when a tag matches nothing or no single
        worker holds them all. Health reporting must never raise - a provider
        whose contexts are missing is precisely the condition being reported,
        so it has to come back as data rather than a 500.

        Args:
            context_tags    - Tags the provider's contexts must all match

        Returns:
            Snapshots of the alive workers that own every tag, in worker id
            order. Empty means this provider's model is loaded on no live
            worker, which is the mis-set worker_ids/tags failure. Also empty
            for an empty tag tuple, since a provider that needs no context
            (remote or debug) has no owning workers to report.
        """
        if not context_tags:
            return []

        matched_per_tag = [
            self._tag_to_context_ids.get(tag, set()) for tag in context_tags
        ]
        # A tag matching no context definition at all cannot be satisfied by
        # any combination, so short circuit before the product.
        if any(not ids for ids in matched_per_tag):
            return []

        owners: set[int] = set()
        for context_ids in product(*matched_per_tag):
            owners |= self._workers_with_contexts(context_ids)

        return [
            self._processes[worker_id].snapshot()
            for worker_id in sorted(owners)
            if self._processes[worker_id].alive
        ]

    def get_context_ids_by_tag(self, tag: str) -> set[int]:
        """
        Gets set of context_ids that have the given tag

        Args:
            tag     - Tag to look up

        Returns:
            Set of context_ids that include this tag; empty if no match
        """
        return set(self._tag_to_context_ids.get(tag, set()))

    def _get_min_utilization(self):
        """
        Gets the worker id with minimum utilization
        """
        min_util = None
        min_util_process = None
        for process_id, process in enumerate(self._processes):
            if min_util is None or min_util > process.utilization:
                min_util = process.utilization
                min_util_process = process_id
        return cast(int, min_util_process)

    def _workers_with_contexts(self, context_ids: tuple[int, ...]) -> set[int]:
        """
        Find workers that own every context in the given group

        Args:
            context_ids     - Group of context_ids that must all live on the same worker

        Returns:
            Set of worker_ids that have all requested contexts initialized
        """
        unique = set(context_ids)
        worker_ids = set(range(len(self._processes)))
        for cid in unique:
            worker_ids &= self._processes_with_context(cid)
            if not worker_ids:
                break
        return worker_ids

    def _processes_with_context(self, context_id: int) -> set[int]:
        """
        Set of worker_ids that have the given context_id pinned to them
        """
        return {
            worker_id
            for worker_id, process in enumerate(self._processes)
            if context_id in process.context_ids
        }

    def _assign_process(
        self, context_tags: tuple[str, ...]
    ) -> tuple[int, tuple[int, ...]]:
        """
        Pick the worker for a job. Contexts are pinned so this just finds workers
        that own every required tag and picks the one with lowest utilization.

        Args:
            context_tags    - Tags the job's contexts must match (in order)

        Returns:
            (worker_id, context_ids tuple matching context_tags order)

        Raises:
            KeyError        if any tag matches no configured context
            RuntimeError    if no single worker holds a context for every tag
        """
        if len(context_tags) == 0:
            return (self._get_min_utilization(), ())

        matched_per_tag: list[set[int]] = []
        for tag in context_tags:
            ids = self._tag_to_context_ids.get(tag, set())
            if not ids:
                raise KeyError(
                    f"context tag: {tag} matched 0 context definitions"
                )
            matched_per_tag.append(ids)

        best_score = None
        best_assignment = None
        for context_ids in product(*matched_per_tag):
            worker_ids = self._workers_with_contexts(context_ids)
            for wid in worker_ids:
                # A dead worker (being replaced by the supervisor) never
                # takes a job: anything queued to it would vanish silently.
                if not self._processes[wid].alive:
                    continue
                score = 1 - self._processes[wid].utilization
                if best_score is None or score > best_score:
                    best_score = score
                    best_assignment = (wid, context_ids)

        if best_assignment is None:
            raise RuntimeError(
                f"No live worker has a context for every tag in: {context_tags}"
            )
        return best_assignment

    def register_job(
        self,
        context_tags: tuple[str, ...],
        period_ms: int,
        job: JobInterface[C, D, R, Conf],
        label: str = "",
        session_uid: str | None = None,
        room_uid: str | None = None,
    ) -> JobHandle[D, R, Conf]:
        """
        Registers a new job with WorkerPool

        Args:
            context_tags    - Tags identifying which contexts the job needs, can be empty
            period_ms       - Frequency at which job should be run
            job             - Definition of job to register
            label           - Opaque grouping label reported to the job observer
            session_uid     - Opaque caller session identifier, forwarded verbatim
                                to WorkerProcessManager.register_job
            room_uid        - Opaque caller room identifier, forwarded verbatim
                                to WorkerProcessManager.register_job

        Returns:
            JobHandle for registered job

        Raises:
            KeyError        if any context_tag matches no configured context
            RuntimeError    if no worker has all required contexts together
        """
        process_id, context_ids = self._assign_process(context_tags)
        return self._processes[process_id].register_job(
            context_ids, period_ms, job, label, session_uid, room_uid
        )

    def shutdown(self):
        """
        Shuts down all WorkerProcesses
        Blocks while waiting for worker processes to exit before returning
        """
        self._shutting_down = True
        if self._supervisor is not None:
            self._supervisor.cancel()
            self._supervisor = None
        for process in self._processes:
            if process.alive:
                process.send_terminate()
        for process in self._processes:
            if process.alive:
                process.wait_shutdown()
