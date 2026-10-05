"""
Unit tests for the pool's supervision of worker processes: a dead worker is
abandoned (its jobs told through JobLostEvent), never routed to, and
replaced by a new manager with the same contexts; the replacement's
context runtime info is what the pool reports.
"""

# pylint: disable=protected-access,unused-argument

import asyncio
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from src.shared.utils.worker_pool import (
    ContextAssignment,
    JobHandle,
    WorkerPool,
    WorkerProcessManager,
)

from .context_definitions import Context


def _manager(mocker: MockerFixture, context_ids: set[int], alive=True):
    """A manager mock with the given contexts and liveness"""
    manager = mocker.MagicMock(spec=WorkerProcessManager)
    manager.context_ids = context_ids
    manager.utilization = 0.0
    manager.alive = alive
    manager.exitcode = None if alive else -9
    manager.context_info = {}
    manager.abandon.return_value = 0
    return manager


@pytest.mark.asyncio
async def test_a_dead_worker_is_abandoned_and_replaced(mocker: MockerFixture):
    """A dead worker is abandoned and replaced"""
    contexts = [ContextAssignment(context_def=Context(0), worker_ids=[0, 1])]
    workers = [_manager(mocker, {0}), _manager(mocker, {0})]
    replacement = _manager(mocker, {0})
    replacement.context_info = {0: {"device": "cpu"}}
    constructed = []

    def build(*args, **kwargs):
        constructed.append((args, kwargs))
        if len(constructed) <= 2:
            return workers[len(constructed) - 1]
        return replacement

    mocker.patch(
        "src.shared.utils.worker_pool.worker_pool.WorkerProcessManager",
        side_effect=build,
    )
    logger = MagicMock()
    pool = WorkerPool(logger, 2, contexts, supervision_interval_sec=0.01)
    try:
        # Worker 1 dies.
        workers[1].alive = False
        for _ in range(100):
            await asyncio.sleep(0.02)
            if pool._processes[1] is replacement:
                break
        assert pool._processes[1] is replacement
        workers[1].abandon.assert_called_once()
        assert pool.worker_restarts == {1: 1}
        assert pool.last_worker_restart_at is not None
        # The replacement was built for worker id 1 with worker 1's contexts
        # and the running loop, on an executor thread.
        args, _ = constructed[2]
        assert args[1] == 1
        assert set(args[2]) == {0}
        assert args[5] is asyncio.get_running_loop()
        assert pool.context_runtime_info(0) == {"device": "cpu"}
        assert pool.context_runtime_info_by_tag("context") == {"device": "cpu"}
        assert any(
            "exited unexpectedly" in call.args[0]
            for call in logger.warning.call_args_list
        )
    finally:
        pool.shutdown()
    # Shutdown only talks to live workers; the dead one is never terminated.
    workers[1].send_terminate.assert_not_called()
    replacement.send_terminate.assert_called_once()


@pytest.mark.asyncio
async def test_routing_skips_a_dead_worker_until_it_is_replaced(
    mocker: MockerFixture,
):
    """Routing skips a dead worker until it is replaced"""
    contexts = [
        ContextAssignment(context_def=Context(0), worker_ids=[0]),
        ContextAssignment(context_def=Context(1, ["diar"]), worker_ids=[1]),
    ]
    workers = [_manager(mocker, {0}), _manager(mocker, {1})]
    mocker.patch(
        "src.shared.utils.worker_pool.worker_pool.WorkerProcessManager",
        side_effect=workers,
    )
    pool = WorkerPool(None, 2, contexts, supervise=False)
    try:
        workers[1].alive = False
        with pytest.raises(RuntimeError, match="No live worker"):
            pool._assign_process(("diar",))
        # Health sees no live owner either; the other worker is unaffected.
        assert pool.load_for_tags(("diar",)) == []
        assert pool._assign_process(("context",)) == (0, (0,))
        workers[1].alive = True
        assert pool._assign_process(("diar",)) == (1, (1,))
    finally:
        pool.shutdown()


@pytest.mark.asyncio
async def test_a_failing_replacement_is_retried(
    mocker: MockerFixture, monkeypatch
):
    """A failing replacement is retried"""
    monkeypatch.setattr(
        "src.shared.utils.worker_pool.worker_pool.WORKER_RESPAWN_RETRY_SEC",
        0.01,
    )
    contexts = [ContextAssignment(context_def=Context(0), worker_ids=[0])]
    dead = _manager(mocker, {0})
    replacement = _manager(mocker, {0})
    attempts = []

    def build(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            return dead
        if len(attempts) == 2:
            raise RuntimeError("model failed to load")
        return replacement

    mocker.patch(
        "src.shared.utils.worker_pool.worker_pool.WorkerProcessManager",
        side_effect=build,
    )
    logger = MagicMock()
    pool = WorkerPool(logger, 1, contexts, supervision_interval_sec=0.01)
    try:
        dead.alive = False
        for _ in range(200):
            await asyncio.sleep(0.02)
            if pool._processes[0] is replacement:
                break
        assert pool._processes[0] is replacement
        assert len(attempts) == 3
        assert any(
            "failed to start" in call.args[0]
            for call in logger.warning.call_args_list
        )
    finally:
        pool.shutdown()


def test_context_defs_for_tag_lists_the_definitions(mocker: MockerFixture):
    """Context defs for tag lists the definitions"""
    first, second = Context(0), Context(1)
    contexts = [
        ContextAssignment(context_def=first, worker_ids=[0]),
        ContextAssignment(context_def=second, worker_ids=[0]),
    ]
    mocker.patch(
        "src.shared.utils.worker_pool.worker_pool.WorkerProcessManager",
        side_effect=[_manager(mocker, {0, 1})],
    )
    pool = WorkerPool(None, 1, contexts, supervise=False)
    try:
        assert pool.context_defs_for_tag("context") == [first, second]
        assert pool.context_defs_for_tag("nope") == []
    finally:
        pool.shutdown()


def test_a_lost_handle_tells_its_owner_once_and_goes_inert():
    """A lost handle tells its owner once and goes inert"""
    queued = []
    handle = JobHandle(3, 7, queued.append, lambda c: None, lambda: None)
    lost = []
    handle.on(handle.JobLostEvent, lost.append)

    handle._mark_lost()

    assert lost == [3]
    handle.queue_data([1])
    handle.deregister()
    assert not queued


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_killed_worker_process_is_replaced_and_jobs_resume():
    """
    End to end with real processes: kill the worker under a registered job,
    the handle learns it was lost, the pool brings a worker with the same
    context back, and a new job registered after that runs.
    """
    # pylint: disable=import-outside-toplevel
    import logging

    from src.shared.logger import ContextLogger

    from .jobs import SumJob

    logger = ContextLogger(logging.getLogger("worker-pool-respawn-test"))
    contexts = [ContextAssignment(context_def=Context(0), worker_ids=[0])]
    pool = WorkerPool(logger, 1, contexts, supervision_interval_sec=0.05)
    try:
        handle = pool.register_job(("context",), 100, SumJob())
        lost = []
        handle.on(handle.JobLostEvent, lost.append)
        old_process = pool._processes[0]
        old_process._process.kill()

        for _ in range(400):
            await asyncio.sleep(0.05)
            if (
                pool._processes[0] is not old_process
                and pool._processes[0].alive
            ):
                break
        assert lost == [0]
        assert pool._processes[0] is not old_process
        assert pool._processes[0].alive
        assert pool.worker_restarts == {0: 1}
        assert pool.load_for_tags(("context",))[0].alive

        results = []
        replacement = pool.register_job(("context",), 100, SumJob())
        replacement.on(replacement.JobResultEvent, results.append)
        replacement.queue_data([1, 2, 3])
        for _ in range(100):
            await asyncio.sleep(0.05)
            if results:
                break
        assert results and not results[0].has_exception
    finally:
        pool.shutdown()


@pytest.mark.asyncio
async def test_a_worker_terminated_by_a_shutdown_signal_is_not_replaced(
    mocker: MockerFixture,
):
    """
    A worker that exited cleanly or on SIGTERM/SIGINT was asked to stop (the
    service is going down); the supervisor leaves it alone instead of
    racing the pool's own shutdown with a replacement.
    """
    contexts = [ContextAssignment(context_def=Context(0), worker_ids=[0])]
    worker = _manager(mocker, {0})
    build = mocker.patch(
        "src.shared.utils.worker_pool.worker_pool.WorkerProcessManager",
        side_effect=[worker],
    )
    pool = WorkerPool(MagicMock(), 1, contexts, supervision_interval_sec=0.01)
    try:
        worker.alive = False
        worker.exitcode = -15
        await asyncio.sleep(0.1)
        assert build.call_count == 1
        worker.abandon.assert_not_called()
        assert not pool.worker_restarts
    finally:
        pool.shutdown()
