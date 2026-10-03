"""
Defines configuration schema for WhisperStreamingProvider
"""

from typing import Optional

from pydantic import BaseModel, TypeAdapter, model_validator


class WhisperStreamingProviderConfig(BaseModel):
    """
    Provider configuration format for WhisperStreamingProvider
    """

    whisper_context_tag: str
    silero_context_tag: str
    job_period_ms: int
    max_buffer_len_sec: float
    local_agree_dim: int

    # `max_buffer_len_sec` used to do three jobs: hard buffer capacity (past
    # which incoming audio is dropped and counted - see
    # WhisperStreamingProviderJob._decode_audio), the tail length that
    # triggers a force-commit-and-purge, and the span handed to Whisper each
    # pass. Split into three so a deployment can bound per-pass transcribe
    # cost independently of how much backlog it tolerates. Both default to
    # `max_buffer_len_sec` when unset, so a config carrying only the old
    # field - or an older service still reading one written by a newer
    # config - behaves exactly as before.
    force_finalize_len_sec: Optional[float] = None
    max_transcribe_len_sec: Optional[float] = None

    vad_detector: bool = False
    vad_threshold: float = 0.5
    vad_neg_threshold: Optional[float] = None
    silence_threshold: float = 0.01
    # Speaker diarization. Fully optional: with `diarization_detector` off the
    # provider registers exactly the job upstream does and none of the fields
    # below are read. With it on, a second job runs on a worker owning the
    # `diarization_context_tag` context, fed the same audio chunks, and labels
    # arrive after the captions (see DiarizationJob / SpeakerLabelAttacher).
    diarization_detector: bool = False
    diarization_context_tag: str = "pyannote_diarization"
    diarization_min_speakers: Optional[int] = None
    diarization_max_speakers: Optional[int] = None
    # Cadence of the diarization job. Defaults to `job_period_ms`. Per-pass
    # cost is set by the window, not the period, so a shorter period lowers
    # label latency at a proportionally higher CPU duty.
    diarization_period_ms: Optional[int] = None
    # Newest audio each pass diarizes. The cost lever: pyannote's embedding
    # stage grows superlinearly with the window (10 s: 0.6 s, 20 s: 5.6 s,
    # 30 s: 11.7 s on the audit's CPU) because every 1 s step of its 10 s
    # segmentation window is embedded. Must exceed the period so consecutive
    # windows overlap and the reconciler can carry speaker identity across.
    diarization_window_sec: float = 10.0
    # The last part of every window is left for the next pass to label, so a
    # word is never attributed from the edge of a window, where segmentation
    # has the least context.
    diarization_edge_margin_sec: float = 0.5
    # A finalized caption still waiting for labels after this long is settled
    # with whatever labels it has, so a stalled diarization job can never
    # leave a caption pending forever.
    diarization_label_timeout_sec: float = 15.0

    # Guard thresholds over Whisper's own quality signals (see
    # TranscriptionJobCounter). Configurable rather than hardcoded so a
    # maintainer can retune them from observed false-positive/negative rates
    # without a code change.
    compression_ratio_guard_threshold: float = 2.4
    avg_logprob_guard_threshold: float = -1.0
    no_speech_prob_guard_threshold: float = 0.6

    @model_validator(mode="after")
    def _resolve_bounded_tail_fields(self) -> "WhisperStreamingProviderConfig":
        """
        Resolves the force-finalize and max-transcribe defaults, then
        enforces the two invariants the bounded-tail split depends on.

        `force_finalize_len_sec` must be at least `max_transcribe_len_sec`:
        the transcribe window slides across whatever the buffer holds
        front-first, and finalization is what advances it. If the tail could
        be force-purged before the window ever reached it, that audio is
        dropped having never been transcribed - silent caption loss, not a
        performance regression, so it is rejected here rather than degrading
        quietly at runtime.

        `job_period_ms` must not exceed `max_buffer_len_sec` in
        milliseconds: a job scheduled less often than the buffer can hold
        guarantees an overflow on every single pass, even with exactly one
        session and no contention.
        """
        if self.force_finalize_len_sec is None:
            self.force_finalize_len_sec = self.max_buffer_len_sec
        if self.max_transcribe_len_sec is None:
            self.max_transcribe_len_sec = self.max_buffer_len_sec

        if self.force_finalize_len_sec < self.max_transcribe_len_sec:
            raise ValueError(
                "force_finalize_len_sec "
                f"({self.force_finalize_len_sec}) must be >= "
                f"max_transcribe_len_sec ({self.max_transcribe_len_sec}) - "
                "otherwise the tail can be force-purged before it is ever "
                "transcribed"
            )

        if self.job_period_ms > self.max_buffer_len_sec * 1000:
            raise ValueError(
                f"job_period_ms ({self.job_period_ms}) must not exceed "
                f"max_buffer_len_sec ({self.max_buffer_len_sec}) in "
                "milliseconds - otherwise every pass overflows the buffer"
            )

        return self

    @model_validator(mode="after")
    def _resolve_diarization_fields(self) -> "WhisperStreamingProviderConfig":
        """
        Resolves the diarization period default and rejects settings that
        would silently produce unlabelled or mislabelled captions: a window
        no longer than the period leaves gaps the reconciler cannot bridge,
        speaker bounds that are not positive or are inverted, and a
        non-positive edge margin or timeout.
        """
        if self.diarization_period_ms is None:
            self.diarization_period_ms = self.job_period_ms
        if not self.diarization_detector:
            return self

        if self.diarization_period_ms <= 0:
            raise ValueError("diarization_period_ms must be positive")
        if self.diarization_window_sec * 1000 <= self.diarization_period_ms:
            raise ValueError(
                f"diarization_window_sec ({self.diarization_window_sec}) must "
                f"exceed diarization_period_ms ({self.diarization_period_ms}) "
                "in seconds - otherwise consecutive windows do not overlap "
                "and speaker identity cannot be carried across them"
            )
        if self.diarization_edge_margin_sec < 0 or (
            self.diarization_edge_margin_sec * 1000
            >= self.diarization_period_ms
        ):
            raise ValueError(
                "diarization_edge_margin_sec must be between 0 and the "
                "diarization period"
            )
        if self.diarization_label_timeout_sec <= 0:
            raise ValueError("diarization_label_timeout_sec must be positive")
        for name in ("diarization_min_speakers", "diarization_max_speakers"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            self.diarization_min_speakers is not None
            and self.diarization_max_speakers is not None
            and self.diarization_min_speakers > self.diarization_max_speakers
        ):
            raise ValueError(
                "diarization_min_speakers must not exceed "
                "diarization_max_speakers"
            )
        return self


whisper_streaming_config_adapter = TypeAdapter[WhisperStreamingProviderConfig](
    WhisperStreamingProviderConfig
)
