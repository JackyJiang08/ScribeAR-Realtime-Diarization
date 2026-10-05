"""
Defines FasterWhisperStreamingProvider
"""

# The provider/session wiring necessarily mirrors the other providers.
# pylint: disable=duplicate-code

import time
from collections import deque
from dataclasses import asdict, dataclass

from src.shared.logger import Logger
from src.shared.utils.diarization_backend import DiarizationContextInterface
from src.shared.utils.speaker_attribution import SpeakerLabelAttacher
from src.shared.utils.speaker_reconciler import SpeakerReconcilerState
from src.shared.utils.worker_pool import (
    JobException,
    JobSuccess,
    WorkerPool,
    is_saturated,
)
from src.transcription_provider_interface import (
    AudioChunkPayload,
    ProviderHealth,
    ProviderKind,
    ProviderStatus,
    TranscriptionJobCounter,
    TranscriptionProviderInterface,
    TranscriptionResult,
    TranscriptionSessionInterface,
)

from .diarization_job import (
    DIARIZATION_JOB_LABEL_SUFFIX,
    DiarizationChunk,
    DiarizationJob,
    DiarizationResult,
)
from .whisper_streaming_config import whisper_streaming_config_adapter
from .whisper_streaming_job import WhisperStreamingProviderJob


@dataclass
class RememberedSpeakers:
    """
    Speaker memory of a session whose socket closed, kept for a reconnect

    Properties:
        state               - The reconciler's exported memory
        next_sequence_id    - The attacher's sequence counter
        expires_at          - Wall clock after which it is dropped
    """

    state: SpeakerReconcilerState
    next_sequence_id: int
    expires_at: float


class WhisperStreamingProvider(TranscriptionProviderInterface):
    """
    TranscriptionProvider that implements WhisperStreaming algorithm
    described in (Macháček et al.)

    @inproceedings{machacek-etal-2023-turning,
        title = "Turning Whisper into Real-Time Transcription System",
        author = "Mach{\'a}{\v{c}}ek, Dominik  and
        Dabre, Raj  and
        Bojar, Ond{\v{r}}ej",
        editor = "Saha, Sriparna  and
        Sujaini, Herry",
        booktitle = "Proceedings of the 13th International Joint Conference on Natural
            Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the
            Association for Computational Linguistics: System Demonstrations",
        month = nov,
        year = "2023",
        address = "Bali, Indonesia",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.ijcnlp-demo.3",
        pages = "17--24",
    }

    Speaker diarization, when enabled, is a second worker-pool job per
    session (`DiarizationJob`) on a worker that owns the pyannote context,
    fed the same audio chunks. Captions are emitted the moment Whisper
    produces them; labels are attached by `SpeakerLabelAttacher` from
    whatever diarization has covered so far and the rest arrive later as
    `SpeakerLabelsEvent`s. With diarization off the provider registers
    exactly the job upstream does and nothing else here runs.
    """

    class _WhisperStreamingSession(TranscriptionSessionInterface):
        """
        Transcription session inferface for WhisperStreamingProvider
        """

        def __init__(
            self,
            provider: "WhisperStreamingProvider",
            logger: Logger,
            session_uid: str | None,
            room_uid: str | None,
        ):
            super().__init__()
            self._log = logger
            self._provider = provider
            # Opaque; stored for a future consumer (Part 2), not read here.
            self.session_uid = session_uid
            self.room_uid = room_uid

            # Not registered here - see _ensure_job. An idle client that never
            # sends audio must never take a worker's job slot, which is also
            # why admission_worker_id below has to tolerate self._job being
            # None.
            self._job = None
            # The diarization job and the attacher exist only when
            # diarization is on; both stay None otherwise so the off path
            # touches nothing upstream does not.
            self._diarization_job = None
            self._attacher: SpeakerLabelAttacher | None = None
            # Set when the diarization job's worker died: the job is gone
            # and a new one is registered on the next audio chunk (the pool
            # is replacing the worker meanwhile). Earliest wall time of the
            # next attempt, so a pool with no live diarization worker yet is
            # asked once a second, not once per chunk.
            self._diarization_retry_at: float | None = None
            # The diarization job's clock as last reported (audio seconds,
            # chunks, wall time of its newest chunk) and the wall times of
            # the chunks this session sent, so a replacement job can start
            # its clock where the lost one stopped (see
            # `_diarization_clock_offset`).
            self._diarization_clock: tuple[float, int, float | None] = (
                0.0,
                0,
                None,
            )
            self._diarization_sent_at: deque[float] = deque(maxlen=4096)
            # Speaker memory to start the diarization job from (a reconnect
            # of a session the provider still remembers) and the newest
            # memory this session's job reported, kept for the next one.
            self._speaker_state = None
            if provider.config.diarization_detector:
                self._attacher = SpeakerLabelAttacher(
                    edge_margin_sec=(
                        provider.config.diarization_edge_margin_sec
                    ),
                    label_timeout_sec=(
                        provider.config.diarization_label_timeout_sec
                    ),
                    attach_gap_sec=provider.config.diarization_attach_gap_sec,
                    revision_margin=(
                        provider.config.diarization_revision_margin
                    ),
                )
                remembered = provider.recall_speakers(session_uid)
                if remembered is not None:
                    self._speaker_state = remembered.state
                    self._attacher.continue_sequence_ids_from(
                        remembered.next_sequence_id
                    )
                    self._log.info(
                        "Continuing speaker labels from an earlier "
                        "connection of this session",
                        context={
                            "session_uid": session_uid,
                            "labels": remembered.state.next_label_id,
                        },
                    )

            self._provider.session_started()

        @property
        def admission_worker_id(self) -> int | None:
            """
            The worker this session's transcription job was actually assigned
            to, read off the handle register_job returned - or None if no job
            has been registered yet

            Read from the JobHandle rather than recomputed, because the pool's
            choice is made from live utilization at registration time and any
            second derivation of it would be a guess about a decision that has
            already been made. This is the only shipped provider that overrides
            it: local ASR compute on a pool worker is exactly what the capacity
            estimator measures.

            None before the first audio chunk is exactly the same "not a
            capacity claim yet" statement the base class makes for a provider
            excluded outright - see
            TranscriptionSessionInterface.admission_worker_id.

            The diarization job is deliberately not part of this answer: it
            lives on a different worker whose load is not what caption
            admission is about.
            """
            return self._job.worker_id if self._job is not None else None

        def _handle_job_result(
            self, result: JobSuccess[TranscriptionResult] | JobException
        ):
            if result.has_exception is True:
                self.emit(self.TranscriptionErrorEvent, result.value)
                return

            value = result.value
            updates = []
            if self._attacher is not None:
                updates = self._attach_known_labels(value, result.counters)

            self._log.info(
                "Completed transcription job",
                context={
                    "stats": asdict(result.stats),
                    "final": (
                        str(value.final) if value.final is not None else None
                    ),
                    "in_progress": (
                        str(value.in_progress)
                        if value.in_progress is not None
                        else None
                    ),
                },
            )
            self.emit(self.TranscriptionResultEvent, value)
            for update in updates:
                self.emit(self.SpeakerLabelsEvent, update)

        def _attach_known_labels(
            self, value: TranscriptionResult, counters: dict[str, float]
        ):
            """
            Labels the result's words from the diarization coverage so far,
            synchronously and without waiting for anything. Words not yet
            covered stay None; finalized sequences get a sequence_id so the
            labels can follow later. Also settles any pending sequence that
            has waited past the timeout.

            Args:
                value       - The caption result about to be emitted
                counters    - The caption job's per-execution counters, read
                                for audio it dropped (which shifts its clock
                                against the diarization job's)

            Returns:
                Settling updates: for a finalized sequence that was fully
                decided at emission but has unattributed words (nothing
                more will come for it), and for pending sequences that
                waited past the timeout
            """
            assert self._attacher is not None
            self._attacher.record_whisper_drop(
                counters.get(
                    TranscriptionJobCounter.AUDIO_DROPPED_BUFFER_FULL_SECONDS,
                    0.0,
                )
            )
            now = time.time()
            updates = []
            if value.final is not None:
                settled = self._attacher.label_sequence(value.final, True, now)
                if settled is not None:
                    updates.append(settled)
            if value.in_progress is not None:
                self._attacher.label_sequence(value.in_progress, False, now)
            return updates + self._attacher.expire(now)

        def _handle_diarization_result(
            self, result: JobSuccess[DiarizationResult | None] | JobException
        ):
            """
            Extends the label coverage with one diarization pass and sends
            the labels any already-shown caption was waiting for

            A diarization failure ends the diarization job (the pool
            deregisters a job that raised) but never the session: captions
            keep flowing without labels, and pending captions settle by
            timeout.
            """
            if result.has_exception is True:
                self._log.warning(
                    "Diarization job failed; captions continue without "
                    "speaker labels",
                    context={"error": str(result.value)},
                )
                self._diarization_job = None
                return
            if result.value is None or self._attacher is None:
                return
            value = result.value
            if value.state is not None:
                self._speaker_state = value.state
            self._diarization_clock = (
                value.audio_received_sec,
                value.chunks_received,
                value.newest_received_at,
            )
            updates = self._attacher.add_coverage(
                value.segments,
                value.window_start,
                value.window_end,
                time.time(),
                confidences=value.confidences,
                relabel=value.relabel,
            )
            for update in updates:
                self.emit(self.SpeakerLabelsEvent, update)

        def _handle_diarization_job_lost(self, worker_id: int) -> None:
            """
            The diarization worker died under this session's job. Captions
            are untouched (they run on another worker); the labels resume
            from the speaker memory the job last reported as soon as a new
            job can be registered, which `handle_audio_chunk` tries on every
            chunk once a second until the pool has the worker back.
            """
            self._log.warning(
                "Diarization worker exited; speaker labels pause until the "
                "pool replaces it, captions continue",
                context={"worker_id": worker_id},
            )
            self._diarization_job = None
            self._diarization_retry_at = time.time()

        def _diarization_clock_offset(self) -> tuple[float, int]:
            """
            Where a replacement diarization job's clock starts: the lost
            job's last reported clock, extended by the chunks this session
            sent after the newest chunk that clock counted, at the lost
            job's mean chunk length. Exact for fixed-size chunks (every
            ScribeAR client sends them), within a chunk otherwise. A first
            registration starts at zero.

            Returns:
                (seconds of audio before the new job's first chunk, chunks
                behind them)
            """
            audio_sec, chunks, newest_at = self._diarization_clock
            if chunks <= 0:
                return 0.0, 0
            sent_after = (
                sum(1 for sent in self._diarization_sent_at if sent > newest_at)
                if newest_at is not None
                else 0
            )
            mean_chunk_sec = audio_sec / chunks
            return audio_sec + sent_after * mean_chunk_sec, chunks + sent_after

        def _register_diarization_job(self) -> None:
            """
            Registers this session's diarization job, seeded with the newest
            speaker memory and, after a loss, the session's audio clock, and
            subscribes to its results and to its loss. Raises what the pool
            raises when no live worker owns the tag.
            """
            assert self._attacher is not None
            clock_offset_sec, chunks_offset = self._diarization_clock_offset()
            job = self._provider.worker_pool.register_job(
                self._provider.diarization_context_tags,
                self._provider.config.diarization_period_ms,
                DiarizationJob(
                    self._provider.config,
                    self._speaker_state,
                    clock_offset_sec=clock_offset_sec,
                    chunks_offset=chunks_offset,
                ),
                self._provider.provider_key + DIARIZATION_JOB_LABEL_SUFFIX,
                session_uid=self.session_uid,
                room_uid=self.room_uid,
            )
            # Subscribed before the result handler on purpose: the tests
            # (and nothing else) read the result handler as the newest
            # subscription on the handle.
            job.on(job.JobLostEvent, self._handle_diarization_job_lost)
            job.on(job.JobResultEvent, self._handle_diarization_result)
            self._diarization_job = job
            self._diarization_retry_at = None

        def _retry_diarization_job(self) -> None:
            """
            Re-registers a diarization job lost with its worker, once the
            retry time has come; a pool that still has no live diarization
            worker postpones the next attempt by a second
            """
            now = time.time()
            if (
                self._diarization_retry_at is None
                or now < self._diarization_retry_at
            ):
                return
            try:
                self._register_diarization_job()
            except (RuntimeError, KeyError) as error:
                self._diarization_retry_at = now + 1.0
                self._log.debug(
                    "Diarization worker not back yet; retrying in 1 s",
                    context={"error": str(error)},
                )
                return
            self._log.info(
                "Diarization job re-registered after its worker was "
                "replaced; speaker labels resume from the session's memory",
                context={"worker_id": self._diarization_job.worker_id},
            )

        def _ensure_job(self) -> None:
            """
            Registers this session's worker-pool job on the first real audio
            chunk, not at construction

            Deferred so an idle client - configured but never streaming - never
            takes a worker's job slot, and so never counts toward that
            worker's live_job_count for capacity admission
            (PLAN-AdmissionControl.md §4). `_admit_registered_job` is called
            immediately after registering, before any data is queued to the
            job, so a refusal can undo the registration and raise before this
            chunk is ever processed - the same "build then undo" shape
            admission used at construction time, just relocated to where
            registration itself now happens. That undo lives on
            TranscriptionSessionInterface rather than here, so all three
            providers get it identically.

            A `register_job` that raises (context tags misconfigured, per
            `describe_health`'s "routed here dies at register_job") now
            surfaces on the first audio chunk instead of at CONFIG time; the
            readiness probe and per-provider health already report that
            misconfiguration independently, so a client still finds out, just
            not until it would have needed the worker anyway.

            The diarization job is registered only after admission passed, so
            a refused session never leaves a diarization job behind on the
            other worker.
            """
            if self._job is not None:
                return

            # Exactly upstream's registration: whisper and silero tags, the
            # upstream job, the provider key as the observer label.
            self._job = self._provider.worker_pool.register_job(
                self._provider.job_context_tags,
                self._provider.config.job_period_ms,
                WhisperStreamingProviderJob(self._provider.config),
                self._provider.provider_key,
                session_uid=self.session_uid,
                room_uid=self.room_uid,
            )
            self._job.on(self._job.JobResultEvent, self._handle_job_result)
            self._admit_registered_job(self._provider, self._log)

            if self._attacher is None:
                return
            self._register_diarization_job()

        def handle_audio_chunk(self, chunk_id: str, chunk: bytes):
            self._ensure_job()
            self._job.queue_data(
                [AudioChunkPayload(chunk_id=chunk_id, audio_bytes=chunk)]
            )
            if self._attacher is None:
                return
            if self._diarization_job is None:
                self._retry_diarization_job()
            now = time.time()
            # Every chunk counts toward the session's audio clock, including
            # the ones sent while the diarization job is being replaced:
            # they are exactly what the replacement's clock must skip.
            self._diarization_sent_at.append(now)
            if self._diarization_job is not None:
                self._diarization_job.queue_data(
                    [DiarizationChunk(chunk_id, chunk, now)]
                )

        def end_session(self):
            super().end_session()
            if self._job is not None:
                self._job.deregister()
            if self._diarization_job is not None:
                self._diarization_job.deregister()
            if self._attacher is not None and self._speaker_state is not None:
                self._provider.remember_speakers(
                    self.session_uid,
                    self._speaker_state,
                    self._attacher.next_sequence_id,
                )
            self._speaker_state = None
            self._provider.session_ended()

    def __init__(
        self,
        provider_config: object,
        logger: Logger,
        worker_pool: WorkerPool,
        provider_key: str,
    ):
        self._log = logger
        self.config = whisper_streaming_config_adapter.validate_python(
            provider_config
        )
        self.worker_pool = worker_pool
        self.provider_key = provider_key
        # Speaker memory of sessions whose socket closed, by session_uid,
        # kept in memory for `diarization_reconnect_grace_sec` so a reconnect
        # continues its labels. Never written anywhere; swept on every
        # remember and recall, and emptied for good when the grace expires.
        self._remembered_speakers: dict[str, RememberedSpeakers] = {}
        if self.config.diarization_detector:
            self._check_diarization_placement()

    def remember_speakers(
        self,
        session_uid: str | None,
        state: SpeakerReconcilerState,
        next_sequence_id: int,
    ) -> None:
        """
        Keeps a closed session's speaker memory for the reconnect grace
        period

        Args:
            session_uid         - The session; None (unknown) is not kept
            state               - The reconciler's exported memory
            next_sequence_id    - The attacher's sequence counter, so the
                                    next connection's ids never collide
        """
        now = time.time()
        self._sweep_remembered(now)
        grace = self.config.diarization_reconnect_grace_sec
        if session_uid is None or grace <= 0:
            return
        self._remembered_speakers[session_uid] = RememberedSpeakers(
            state=state,
            next_sequence_id=next_sequence_id,
            expires_at=now + grace,
        )

    def recall_speakers(
        self, session_uid: str | None
    ) -> "RememberedSpeakers | None":
        """
        Takes the speaker memory remembered for a session, if any is left
        within its grace period. The memory is handed over, not copied: it
        belongs to the new connection now

        Args:
            session_uid - The session reconnecting

        Returns:
            The remembered memory, or None
        """
        now = time.time()
        self._sweep_remembered(now)
        if session_uid is None:
            return None
        return self._remembered_speakers.pop(session_uid, None)

    @property
    def remembered_sessions(self) -> int:
        """
        Closed sessions whose speaker memory is still within its grace
        """
        return len(self._remembered_speakers)

    def _sweep_remembered(self, now: float) -> None:
        expired = [
            uid
            for uid, entry in self._remembered_speakers.items()
            if entry.expires_at <= now
        ]
        for uid in expired:
            del self._remembered_speakers[uid]

    def _check_diarization_placement(self) -> None:
        """
        Fails fast when no worker owns the diarization context, and warns
        when the only workers that do also run captions

        A missing context used to surface as a `RuntimeError` on the first
        audio chunk, a 1011 close and a node-server reconnect loop. A shared
        worker is legal but defeats the point: the worker runs one job at a
        time, so every diarization pass holds the caption job up for its
        duration, and the context's `nice` lowers the caption job's priority
        along with it. Said once at start-up, where a deployer reads it.
        """
        diarization_workers = {
            worker.worker_id
            for worker in self.worker_pool.load_for_tags(
                self.diarization_context_tags
            )
        }
        if not diarization_workers:
            raise ValueError(
                f"diarization_detector is on for provider "
                f"'{self.provider_key}' but no live worker owns a context "
                f"tagged '{self.config.diarization_context_tag}'; add the "
                "pyannote-diarization context to provider_config.json with "
                "worker_ids of its own, or turn diarization_detector off"
            )
        tag = self.config.diarization_context_tag
        definitions = self.worker_pool.context_defs_for_tag(tag)
        if definitions and not any(
            isinstance(definition, DiarizationContextInterface)
            for definition in definitions
        ):
            kinds = sorted({type(d).__name__ for d in definitions})
            raise ValueError(
                f"diarization_context_tag '{tag}' of provider "
                f"'{self.provider_key}' resolves to {kinds}, which cannot "
                "diarize; point it at the pyannote-diarization context's "
                "tag (its `tags` in provider_config.json)"
            )
        caption_workers = {
            worker.worker_id
            for worker in self.worker_pool.load_for_tags(self.job_context_tags)
        }
        if diarization_workers <= caption_workers:
            self._log.warning(
                "Every worker owning the diarization context also runs "
                "captions; diarization passes will delay captions on that "
                "worker. Give the pyannote-diarization context worker_ids of "
                "its own (num_workers >= 2).",
                context={
                    "provider_key": self.provider_key,
                    "diarization_workers": sorted(diarization_workers),
                    "caption_workers": sorted(caption_workers),
                },
            )

    @property
    def job_period_ms(self) -> int | None:
        # The same field the session passes to register_job, read from the same
        # config object, so the reported period cannot drift from the scheduled
        # one.
        return self.config.job_period_ms

    @property
    def context_tags(self) -> list[str]:
        # The registry resolves this tag against its tag-to-device map to
        # report `providerDevice` on /metrics/status — the provider itself
        # does not have access to the context's config, so the registry does
        # the lookup rather than the provider.
        return [self.config.whisper_context_tag]

    @property
    def job_context_tags(self) -> tuple[str, ...]:
        """
        Every context tag a session's caption job needs: whisper and silero,
        exactly as upstream registers it. Diarization never joins this tuple;
        its context lives on another worker and is routed separately
        (`diarization_context_tags`).
        """
        return (self.config.whisper_context_tag, self.config.silero_context_tag)

    @property
    def diarization_context_tags(self) -> tuple[str, ...]:
        """
        The context tag the diarization job is routed by, or an empty tuple
        when diarization is off
        """
        if not self.config.diarization_detector:
            return ()
        return (self.config.diarization_context_tag,)

    def create_session(
        self,
        session_config: object,
        session_uid: str | None,
        room_uid: str | None,
        logger: Logger,
    ):
        # Session config was already unused before session_uid/room_uid
        # existed; accepted for interface parity only.
        del session_config
        return self._WhisperStreamingSession(
            self, logger, session_uid, room_uid
        )

    async def describe_health(self):
        """
        Gets health of this local model provider

        Reads in-memory worker state only - no I/O, so polling this cannot
        perturb transcription.

        The failure this exists to catch: `worker_ids`/`tags` misconfigured so
        that no live worker owns both the whisper and silero contexts. The pool
        is then perfectly healthy and readiness returns 200, but every session
        routed here dies at `register_job`. Nothing detected that before B1.7.

        With diarization on, a missing diarization context is reported the
        same way (DOWN, naming the tag). A saturated diarization worker is
        only a note: captions are unaffected, labels lag.
        """
        tags = self.job_context_tags
        owning_workers = self.worker_pool.load_for_tags(tags)
        model_loaded = len(owning_workers) > 0
        diarization_workers = (
            self.worker_pool.load_for_tags(self.diarization_context_tags)
            if self.config.diarization_detector
            else []
        )
        diarization_loaded = (
            not self.config.diarization_detector or len(diarization_workers) > 0
        )

        if not model_loaded:
            status = ProviderStatus.DOWN
            detail = (
                f"no live worker owns contexts {tags}; check worker_ids and "
                "tags in provider_config"
            )
        elif not diarization_loaded:
            status = ProviderStatus.DOWN
            detail = (
                "no live worker owns the diarization context "
                f"{self.diarization_context_tags}; check worker_ids and tags "
                "in provider_config or turn diarization_detector off"
            )
        elif all(is_saturated(w) for w in owning_workers):
            # Every worker that could take this provider's work is pinned. The
            # provider still transcribes, just behind realtime.
            status = ProviderStatus.DEGRADED
            detail = (
                f"all {len(owning_workers)} workers serving this provider are "
                "saturated; transcription will fall behind realtime"
            )
        else:
            status = ProviderStatus.OK
            detail = None
            if diarization_workers and all(
                is_saturated(w) for w in diarization_workers
            ):
                detail = (
                    "the diarization worker is saturated; captions are "
                    "unaffected but speaker labels will lag"
                )

        return ProviderHealth(
            kind=ProviderKind.LOCAL,
            status=status,
            active_sessions=self.active_sessions,
            # The model name lives on the context config, not here - this
            # provider only holds the tag that routes to it.
            model=None,
            model_loaded=model_loaded and diarization_loaded,
            owning_workers=owning_workers,
            detail=detail,
        )

    def cleanup_provider(self):
        pass
