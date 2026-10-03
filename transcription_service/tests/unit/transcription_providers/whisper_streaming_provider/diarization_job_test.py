"""
Unit tests for DiarizationJob: it diarizes only the newest window, never
queues audio, counts what it skipped, and reports lag.
"""

# pylint: disable=protected-access

import io
import logging
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from src.shared.utils.speaker_reconciler import SpeakerSegment
from src.transcription_provider_interface import TranscriptionJobCounter
from src.transcription_providers.whisper_streaming_provider.diarization_job import (
    SAMPLE_RATE,
    DiarizationChunk,
    DiarizationJob,
)
from src.transcription_providers.whisper_streaming_provider.whisper_streaming_config import (
    WhisperStreamingProviderConfig,
)


def make_job(**overrides) -> DiarizationJob:
    """A job with a 10 s window and diarization on."""
    config = {
        "whisper_context_tag": "w",
        "silero_context_tag": "s",
        "job_period_ms": 5000,
        "max_buffer_len_sec": 30,
        "local_agree_dim": 2,
        "diarization_detector": True,
        "diarization_window_sec": 10.0,
    }
    config.update(overrides)
    return DiarizationJob(WhisperStreamingProviderConfig(**config))


def chunk(seconds: float, chunk_id: str = "a", received_at=None):
    """`seconds` of silence as a WAV chunk payload."""
    samples = np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return DiarizationChunk(
        chunk_id, buf.getvalue(), received_at or time.time()
    )


@pytest.fixture(name="log")
def log_fixture():
    """A logger stub."""
    return MagicMock(spec=logging.Logger)


def _diarizer(segments):
    """A pyannote stand-in returning window-relative segments."""
    diarizer = MagicMock()
    diarizer.diarize.return_value = segments
    return diarizer


def test_skips_the_pass_when_no_audio_arrived(log):
    """An empty period costs nothing and reports nothing."""
    job = make_job()
    diarizer = _diarizer([])

    assert job.process_batch(log, (diarizer,), []) is None
    diarizer.diarize.assert_not_called()


def test_skips_the_pass_while_the_buffer_is_too_short(log):
    """Under two seconds of audio is not worth a pass."""
    job = make_job()
    diarizer = _diarizer([])

    assert job.process_batch(log, (diarizer,), [chunk(1.0)]) is None
    diarizer.diarize.assert_not_called()


def test_labels_are_reported_on_the_session_timeline(log):
    """
    Window-relative segments from the pipeline come back offset by the
    window start, with the reconciler's stable labels, and the window bounds
    say what is now decided.
    """
    job = make_job()
    diarizer = _diarizer([SpeakerSegment(1.0, 2.0, "SPEAKER_00")])
    job.process_batch(log, (diarizer,), [chunk(5.0, "a")])

    # 15 s received in total: the window is the newest 10 s, starting at 5 s.
    result = job.process_batch(log, (diarizer,), [chunk(10.0, "b")])

    assert result is not None
    assert result.window_start == pytest.approx(5.0)
    assert result.window_end == pytest.approx(15.0)
    # The first pass minted spk_0 at 1-2 s; this raw label overlaps nothing
    # from it, so the reconciler mints a fresh session label.
    assert result.segments == [SpeakerSegment(6.0, 7.0, "spk_1")]
    args, _ = diarizer.diarize.call_args
    assert len(args[0]) == 10 * SAMPLE_RATE


def test_audio_that_no_longer_fits_the_window_is_skipped_and_counted(log):
    """
    Back-pressure is by skipping: when more than a window arrives between
    passes the oldest audio is dropped unlabelled and counted, never queued.
    """
    job = make_job()
    diarizer = _diarizer([])
    job.process_batch(log, (diarizer,), [chunk(5.0, "a")])
    job.drain_counters()

    # 20 s arrive at once: the newest 10 s are kept; 5 s of the old window
    # was already covered, so 10 s of this batch were never looked at.
    job.process_batch(log, (diarizer,), [chunk(20.0, "b")])

    counters = job.drain_counters()
    assert counters[
        TranscriptionJobCounter.DIARIZATION_UNCOVERED_SECONDS
    ] == pytest.approx(10.0)
    assert counters[
        TranscriptionJobCounter.DIARIZATION_AUDIO_SECONDS
    ] == pytest.approx(20.0)


def test_counts_runs_seconds_and_lag(log):
    """One pass counts a run, its wall time, and how old its newest audio was."""
    job = make_job()
    diarizer = _diarizer([])

    result = job.process_batch(
        log, (diarizer,), [chunk(5.0, "a", received_at=time.time() - 2.0)]
    )

    counters = job.drain_counters()
    assert counters[TranscriptionJobCounter.DIARIZATION_RUNS] == 1
    assert counters[TranscriptionJobCounter.DIARIZATION_SECONDS] >= 0
    assert counters[TranscriptionJobCounter.DIARIZATION_LAG_SECONDS] >= 2.0
    assert result is not None and result.lag_sec >= 2.0


def test_a_failed_pass_is_counted_and_returns_nothing(log):
    """A pipeline error costs that pass's labels, not the job."""
    job = make_job()
    diarizer = MagicMock()
    diarizer.diarize.side_effect = RuntimeError("boom")

    assert job.process_batch(log, (diarizer,), [chunk(5.0)]) is None

    counters = job.drain_counters()
    assert counters[TranscriptionJobCounter.DIARIZATION_FAILED] == 1
    assert TranscriptionJobCounter.DIARIZATION_RUNS not in counters


def test_labels_minted_are_counted_once(log):
    """A new raw label mints one session label, reported once."""
    job = make_job()
    diarizer = _diarizer(
        [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(2.0, 4.0, "SPEAKER_01"),
        ]
    )

    job.process_batch(log, (diarizer,), [chunk(5.0)])

    counters = job.drain_counters()
    assert counters[TranscriptionJobCounter.DIARIZATION_LABELS_MINTED] == 2
    assert job.labels_minted == 2
