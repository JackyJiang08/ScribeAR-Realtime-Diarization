"""
Defines DiarizationJob, the worker-pool job that labels speakers beside,
and never in front of, the caption job
"""

import time
from dataclasses import dataclass, field

import numpy as np

from src.shared.logger import Logger
from src.shared.utils.audio_decoder import AudioDecoder, TargetFormat
from src.shared.utils.np_circular_buffer import NPCircularBuffer
from src.shared.utils.speaker_reconciler import (
    SpeakerReconciler,
    SpeakerReconcilerConfig,
    SpeakerReconcilerState,
    SpeakerSegment,
)
from src.shared.utils.worker_pool import JobInterface
from src.transcription_contexts.pyannote_diarization_context import (
    PyannoteDiarizationModelType,
)
from src.transcription_provider_interface import (
    JobCounterCollector,
    TranscriptionClientError,
    TranscriptionJobCounter,
)

from .whisper_streaming_config import WhisperStreamingProviderConfig

SAMPLE_RATE = 16000
NUM_CHANNELS = 1

#: Suffix the diarization job's observer label carries, appended to the
#: provider key the caption job is labelled with. The metrics registry keys
#: on it to keep diarization executions out of every `asr_*` series: the
#: caption numbers on the dashboard must describe the caption job alone, and
#: a diarization pass that overruns must show up as diarization lag, never
#: as a dropped caption period.
DIARIZATION_JOB_LABEL_SUFFIX = ":diarization"

# Audio shorter than this is not worth a pass: pyannote pads its 10 s
# segmentation window and the labels on a fraction of a second of speech are
# noise. The next pass, a period later, covers it with context.
MIN_WINDOW_SEC = 2.0


@dataclass
class DiarizationChunk:
    """
    One source audio chunk as the diarization job receives it

    Properties:
        chunk_id    - Correlation id of the chunk, as the caption job gets it
        audio_bytes - The chunk, decodable on its own (WAV)
        received_at - `time.time()` when the service received the chunk, so
                        the job can report how old the audio it just
                        labelled is (the diarization lag)
    """

    chunk_id: str
    audio_bytes: bytes
    received_at: float


@dataclass
class DiarizationResult:
    """
    Speaker labels for the newest window of a session's audio

    Properties:
        segments        - Reconciled, session-wide speaker labels over the
                            window, session timeline (seconds since the first
                            chunk, counting every chunk received)
        window_start    - First instant the pass looked at
        window_end      - Last instant the pass looked at; everything before
                            it minus the attacher's edge margin is now decided
        lag_sec         - Age of the newest audio in the window when the
                            labels were ready
        labels_minted   - Session labels minted so far, cumulative
        confidences     - Session label -> score the reconciler attached it
                            with in this pass (cosine similarity to the
                            speaker's centroid, plus overlap bonus)
        relabel         - Junior -> senior labels merged in this pass
        state           - The reconciler's speaker memory after this pass,
                            when it changed; the session keeps the newest
                            one so a reconnect can continue the labels
    """

    segments: list[SpeakerSegment]
    window_start: float
    window_end: float
    lag_sec: float
    labels_minted: int
    confidences: dict[str, float] = field(default_factory=dict)
    relabel: dict[str, str] = field(default_factory=dict)
    state: SpeakerReconcilerState | None = None


def reconciler_config_from(
    config: WhisperStreamingProviderConfig,
) -> SpeakerReconcilerConfig:
    """
    The reconciler thresholds a provider config asks for
    """
    return SpeakerReconcilerConfig(
        match_threshold=config.diarization_match_threshold,
        new_speaker_threshold=config.diarization_new_speaker_threshold,
        attach_threshold=config.diarization_attach_threshold,
        min_mint_duration_sec=config.diarization_min_mint_sec,
        max_speakers=config.diarization_max_session_speakers,
        merge_threshold=config.diarization_merge_threshold,
        overlap_bonus=config.diarization_overlap_bonus,
        sustained_split_sec=config.diarization_sustained_split_sec,
        recluster_period_sec=config.diarization_recluster_period_sec,
    )


class DiarizationJob(
    JobInterface[
        tuple[PyannoteDiarizationModelType],
        DiarizationChunk,
        DiarizationResult | None,
        None,
    ]
):
    """
    Diarizes the newest `diarization_window_sec` of a session's audio once
    per `diarization_period_ms`, in whatever worker owns the pyannote context

    The caption job never waits for this one: the two are separate pool jobs
    fed the same chunks, and labels are attached to captions afterwards by
    the session (`SpeakerLabelAttacher`). Back-pressure is by skipping, never
    by queueing: the buffer holds at most one window, so audio that arrived
    while a pass was running and no longer fits is purged unlabelled and
    counted (`DIARIZATION_UNCOVERED_SECONDS`), and a period the pass overran
    is dropped by the pool like any other job's. Both are exported, so a
    deployment can see labels falling behind without captions ever doing so.
    """

    def __init__(
        self,
        config: WhisperStreamingProviderConfig,
        state: SpeakerReconcilerState | None = None,
    ):
        """
        Args:
            config  - The provider config
            state   - Speaker memory of an earlier connection of the same
                        session to continue from (reconnect), or None
        """
        self._counters = JobCounterCollector()
        self._decoder = AudioDecoder(
            SAMPLE_RATE, NUM_CHANNELS, TargetFormat.FLOAT_32
        )
        self._window_samples = int(SAMPLE_RATE * config.diarization_window_sec)
        # Exactly one window: nothing older than a window is ever useful to
        # this job, so there is nothing to queue.
        self._buffer = NPCircularBuffer(self._window_samples, dtype=np.float32)
        # Absolute sample index of buffer[0], on a timeline that counts every
        # chunk this job ever received - the timeline the labels are reported
        # on.
        self._buffer_offset_samples = 0
        self._total_samples = 0
        # End of the newest window diarized so far, absolute samples. Audio
        # purged from the buffer beyond this point was never labelled.
        self._covered_through_samples = 0
        self._newest_received_at: float | None = None
        self._min_speakers = config.diarization_min_speakers
        self._max_speakers = config.diarization_max_speakers
        self._reconciler = SpeakerReconciler(
            config=reconciler_config_from(config), state=state
        )
        self._recluster = config.diarization_recluster_period_sec > 0
        self._state_version_sent = self._reconciler.state_version

    @property
    def labels_minted(self) -> int:
        """
        Session labels the reconciler has minted so far
        """
        return self._reconciler.labels_minted

    def _decode_audio(self, batch: list[DiarizationChunk]) -> int:
        """
        Decodes the batch into the window buffer, keeping the newest audio

        Returns:
            Samples appended
        """
        appended = 0
        for chunk in batch:
            try:
                samples = self._decoder.decode(chunk.audio_bytes)
            except ValueError as error:
                raise TranscriptionClientError(str(error)) from error

            # Counted on everything received, before any of it is skipped,
            # so the RTF denominator is the audio the session produced and
            # not the audio this job chose to look at.
            self._counters.inc(
                TranscriptionJobCounter.DIARIZATION_AUDIO_SECONDS,
                len(samples) / SAMPLE_RATE,
            )
            if len(samples) > self._window_samples:
                # A single chunk longer than the window: only its tail can
                # ever be labelled. Everything already buffered and the head
                # of this chunk are skipped, and counted where no pass ever
                # saw them.
                skipped = len(samples) - self._window_samples
                self._purge_oldest(len(self._buffer))
                self._skip_samples(skipped)
                samples = samples[-self._window_samples :]
            free = self._window_samples - len(self._buffer)
            if len(samples) > free:
                self._purge_oldest(len(samples) - free)
            self._buffer.append(samples)
            self._total_samples += len(samples)
            appended += len(samples)
            self._newest_received_at = chunk.received_at
        return appended

    def _purge_oldest(self, amount: int) -> None:
        """
        Drops the oldest `amount` samples from the window and counts the ones
        no pass ever looked at
        """
        if amount <= 0:
            return
        purge_end = self._buffer_offset_samples + amount
        uncovered = max(0, purge_end - self._covered_through_samples)
        if uncovered > 0:
            self._counters.inc(
                TranscriptionJobCounter.DIARIZATION_UNCOVERED_SECONDS,
                uncovered / SAMPLE_RATE,
            )
            self._covered_through_samples = purge_end
        self._buffer.purge(amount)
        self._buffer_offset_samples = purge_end

    def _skip_samples(self, amount: int) -> None:
        """
        Advances the timeline past `amount` samples that never enter the
        buffer, counting them as uncovered: nothing will ever label them
        """
        self._total_samples += amount
        self._counters.inc(
            TranscriptionJobCounter.DIARIZATION_UNCOVERED_SECONDS,
            amount / SAMPLE_RATE,
        )
        self._covered_through_samples = self._total_samples
        self._buffer_offset_samples = self._total_samples

    def process_batch(
        self,
        log: Logger,
        contexts: tuple[PyannoteDiarizationModelType],
        batch: list[DiarizationChunk],
    ) -> DiarizationResult | None:
        (diarizer,) = contexts
        if self._reconciler.clusterer is None and self._recluster:
            # The clustering the model ships, for the periodic session-level
            # re-clustering; resolved on the first batch because the context
            # only exists inside the worker
            self._reconciler.clusterer = getattr(
                diarizer, "track_clusterer", None
            )
            self._recluster = self._reconciler.clusterer is not None

        appended = self._decode_audio(batch)
        if appended == 0:
            # No new audio since the previous pass: nothing to label.
            return None
        if len(self._buffer) < MIN_WINDOW_SEC * SAMPLE_RATE:
            return None

        window = np.asarray(self._buffer.get())
        window_start_samples = self._buffer_offset_samples
        window_end_samples = window_start_samples + len(window)

        diarized = self._run_pass(log, diarizer, window)
        if diarized is None:
            return None

        offset_sec = window_start_samples / SAMPLE_RATE
        reconciled = self._reconcile(
            [
                SpeakerSegment(
                    start=offset_sec + segment.start,
                    end=offset_sec + segment.end,
                    speaker=segment.speaker,
                )
                for segment in diarized.segments
            ],
            diarized.embeddings,
        )

        self._covered_through_samples = max(
            self._covered_through_samples, window_end_samples
        )
        lag_sec = (
            max(0.0, time.time() - self._newest_received_at)
            if self._newest_received_at is not None
            else 0.0
        )
        self._counters.inc(
            TranscriptionJobCounter.DIARIZATION_LAG_SECONDS, lag_sec
        )

        state = None
        if self._reconciler.state_version != self._state_version_sent:
            state = self._reconciler.export_state()
            self._state_version_sent = self._reconciler.state_version

        return DiarizationResult(
            segments=reconciled,
            window_start=offset_sec,
            window_end=window_end_samples / SAMPLE_RATE,
            lag_sec=lag_sec,
            labels_minted=self._reconciler.labels_minted,
            confidences=dict(self._reconciler.last_confidence),
            relabel=dict(self._reconciler.last_merges),
            state=state,
        )

    def _run_pass(self, log: Logger, diarizer, window: np.ndarray):
        """
        One pyannote pass over the window, counted; None when it raised
        """
        started = time.perf_counter()
        try:
            diarized = diarizer.diarize(
                window,
                SAMPLE_RATE,
                min_speakers=self._min_speakers,
                max_speakers=self._max_speakers,
            )
        except Exception as error:  # pylint: disable=broad-exception-caught
            self._counters.inc(TranscriptionJobCounter.DIARIZATION_FAILED)
            self._counters.inc(
                TranscriptionJobCounter.DIARIZATION_SECONDS,
                time.perf_counter() - started,
            )
            log.warning(f"Diarization failed: {error}", exc_info=error)
            return None
        self._counters.inc(TranscriptionJobCounter.DIARIZATION_RUNS)
        self._counters.inc(
            TranscriptionJobCounter.DIARIZATION_SECONDS,
            time.perf_counter() - started,
        )
        return diarized

    def _reconcile(
        self, session_relative: list[SpeakerSegment], embeddings: dict
    ) -> list[SpeakerSegment]:
        """
        Maps the pass's raw labels onto session labels, counting the cost
        and the labels minted
        """
        started = time.perf_counter()
        minted_before = self._reconciler.labels_minted
        reconciled = self._reconciler.reconcile(session_relative, embeddings)
        self._counters.inc(
            TranscriptionJobCounter.RECONCILER_SECONDS,
            time.perf_counter() - started,
        )
        minted = self._reconciler.labels_minted - minted_before
        if minted > 0:
            self._counters.inc(
                TranscriptionJobCounter.DIARIZATION_LABELS_MINTED, minted
            )
        return reconciled

    def drain_counters(self) -> dict[str, float]:
        return self._counters.drain()

    def update_config(self, log: Logger, contexts: tuple, config: None) -> None:
        raise TranscriptionClientError("On the fly config update not supported")
