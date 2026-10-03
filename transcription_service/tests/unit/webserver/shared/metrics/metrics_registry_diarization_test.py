"""
Unit tests for how MetricsRegistry keeps the diarization job's executions out
of the caption series and in their own.
"""

from src.shared.utils.worker_pool import (
    DROPPED_PERIODS_COUNTER,
    JobExecutionObservation,
    JobStatistics,
)
from src.transcription_provider_interface import TranscriptionJobCounter
from src.webserver.shared.metrics import MetricsRegistry
from src.webserver.shared.metrics.metrics_registry import (
    DIARIZATION_JOB_LABEL_SUFFIX,
)

NS_PER_MS = 1_000_000
WHISPER = {"provider_key": "whisper"}


def make_observation(
    label: str,
    execution_ms: float = 1200,
    exception: Exception | None = None,
    counters: dict[str, float] | None = None,
) -> JobExecutionObservation:
    """An observation with the given execution time and counters."""
    return JobExecutionObservation(
        worker_id=1,
        job_id=7,
        label=label,
        stats=JobStatistics(
            period_start_ns=0,
            job_scheduled_time_ns=0,
            start_execute_time_ns=0,
            complete_time_ns=int(execution_ms * NS_PER_MS),
        ),
        exception=exception,
        counters=counters or {},
    )


def test_diarization_execution_never_touches_the_caption_series():
    """
    A diarization pass must not count as a caption job, a caption execution
    time, a caption RTF sample or a dropped caption period - those series
    describe the caption job alone.
    """
    registry = MetricsRegistry()

    registry.record_job_execution(
        make_observation(
            "whisper" + DIARIZATION_JOB_LABEL_SUFFIX,
            counters={
                TranscriptionJobCounter.DIARIZATION_RUNS: 1,
                TranscriptionJobCounter.DIARIZATION_AUDIO_SECONDS: 5.0,
                DROPPED_PERIODS_COUNTER: 2,
            },
        )
    )

    assert registry.jobs_completed_total.entries() == []
    assert registry.asr_execution_ms.series_labels() == []
    assert registry.asr_rtf.series_labels() == []
    assert registry.asr_dropped_periods_total.entries() == []


def test_diarization_execution_is_folded_under_the_caption_provider_key():
    """
    The suffix is stripped so the diarization series carry the same
    provider_key as the captions they label, which is how a dashboard joins
    the two.
    """
    registry = MetricsRegistry()

    registry.record_job_execution(
        make_observation(
            "whisper" + DIARIZATION_JOB_LABEL_SUFFIX,
            execution_ms=1200,
            counters={
                TranscriptionJobCounter.DIARIZATION_RUNS: 1,
                TranscriptionJobCounter.DIARIZATION_SECONDS: 1.1,
                TranscriptionJobCounter.DIARIZATION_LABELS_MINTED: 2,
                TranscriptionJobCounter.DIARIZATION_AUDIO_SECONDS: 4.0,
                TranscriptionJobCounter.DIARIZATION_LAG_SECONDS: 1.5,
                TranscriptionJobCounter.DIARIZATION_UNCOVERED_SECONDS: 0.5,
                DROPPED_PERIODS_COUNTER: 2,
            },
        )
    )

    assert registry.diarization_runs_total.entries()[0].labels == WHISPER
    assert registry.diarization_runs_total.entries()[0].value == 1
    assert registry.diarization_labels_minted_total.entries()[0].value == 2
    assert registry.diarization_audio_seconds_total.entries()[0].value == 4.0
    assert (
        registry.diarization_uncovered_seconds_total.entries()[0].value == 0.5
    )
    assert registry.diarization_dropped_periods_total.entries()[0].value == 2
    execution = registry.diarization_execution_ms.summary(WHISPER)
    assert execution is not None and execution.maximum == 1200
    lag = registry.diarization_lag_ms.summary(WHISPER)
    assert lag is not None and lag.maximum == 1500
    # 1.2 s of compute for 4 s of audio
    rtf = registry.diarization_rtf.summary(WHISPER)
    assert rtf is not None and abs(rtf.maximum - 0.3) < 1e-9


def test_a_raised_diarization_pass_counts_as_a_failed_pass():
    """A job that raised is a failed pass on the diarization series only."""
    registry = MetricsRegistry()

    registry.record_job_execution(
        make_observation(
            "whisper" + DIARIZATION_JOB_LABEL_SUFFIX,
            exception=RuntimeError("boom"),
        )
    )

    assert registry.diarization_failed_total.entries()[0].value == 1
    assert registry.jobs_failed_total.entries() == []


def test_caption_execution_cannot_report_a_diarization_counter():
    """
    A caption job reporting a diarization counter name is ignored, so the
    two sets of series can never bleed into each other.
    """
    registry = MetricsRegistry()

    registry.record_job_execution(
        make_observation(
            "whisper",
            counters={
                TranscriptionJobCounter.DIARIZATION_RUNS: 1,
                TranscriptionJobCounter.AUDIO_SECONDS_DECODED: 5.0,
            },
        )
    )

    assert registry.diarization_runs_total.entries() == []
    assert registry.jobs_completed_total.entries()[0].labels == WHISPER
    assert registry.asr_audio_seconds_total.entries()[0].value == 5.0
