"""
Unit tests for the diarization wiring in WhisperStreamingProvider: with
diarization on a session runs two pool jobs fed the same audio, captions are
emitted immediately with whatever labels are known, late labels arrive as
SpeakerLabelsEvents, and with diarization off nothing beyond upstream runs.
"""

# pylint: disable=protected-access

import time
from unittest.mock import MagicMock

import pytest

from src.shared.logger import Logger
from src.shared.utils.speaker_reconciler import (
    SpeakerReconcilerState,
    SpeakerSegment,
)
from src.shared.utils.worker_pool import (
    JobException,
    JobStatistics,
    JobSuccess,
    WorkerPool,
    WorkerSnapshot,
)
from src.transcription_provider_interface import (
    ProviderStatus,
    TranscriptionJobCounter,
    TranscriptionResult,
    TranscriptionSequence,
    TranscriptionSessionInterface,
)
from src.transcription_providers.whisper_streaming_provider import (
    WhisperStreamingProvider,
)
from src.transcription_providers.whisper_streaming_provider.diarization_job import (
    DIARIZATION_JOB_LABEL_SUFFIX,
    DiarizationChunk,
    DiarizationJob,
    DiarizationResult,
)
from src.transcription_providers.whisper_streaming_provider.whisper_streaming_job import (
    WhisperStreamingProviderJob,
)

WHISPER_TAG = "whisper_context"
SILERO_TAG = "silero_context"
PYANNOTE_TAG = "pyannote_diarization"
PROVIDER_KEY = "whisper"

BASE_CONFIG = {
    "whisper_context_tag": WHISPER_TAG,
    "silero_context_tag": SILERO_TAG,
    "job_period_ms": 5000,
    "max_buffer_len_sec": 30.0,
    "local_agree_dim": 2,
}
DIARIZED_CONFIG = {
    **BASE_CONFIG,
    "diarization_detector": True,
    "diarization_context_tag": PYANNOTE_TAG,
    "diarization_edge_margin_sec": 0.5,
}

STATS = JobStatistics(0, 0, 0, 1)


def _snapshot(worker_id: int) -> WorkerSnapshot:
    return WorkerSnapshot(
        worker_id=worker_id,
        utilization=0.1,
        live_job_count=1,
        total_jobs_registered=1,
        context_ids={0},
        alive=True,
        active_jobs=(),
    )


@pytest.fixture(name="mock_logger")
def mock_logger_fixture():
    """A logger stub."""
    return MagicMock(spec=Logger)


@pytest.fixture(name="mock_worker_pool")
def mock_worker_pool_fixture():
    """
    A pool with captions on worker 0 and diarization on worker 1, handing
    out a distinct job handle per registration.
    """
    pool = MagicMock(spec=WorkerPool)

    def load_for_tags(tags):
        if tags == (WHISPER_TAG, SILERO_TAG):
            return [_snapshot(0)]
        if tags == (PYANNOTE_TAG,):
            return [_snapshot(1)]
        return []

    pool.load_for_tags.side_effect = load_for_tags
    pool.register_job.side_effect = lambda *args, **kwargs: MagicMock(
        worker_id=1 if isinstance(args[2], DiarizationJob) else 0
    )
    return pool


def _emit_result(session, handle, value, counters=None):
    """Fires a job result through the handler the session registered."""
    del session  # the handle carries the handler; kept for readability
    callback = handle.on.call_args[0][1]
    callback(JobSuccess(value, STATS, counters or {}))


def test_diarization_off_registers_exactly_the_upstream_job(
    mock_logger, mock_worker_pool
):
    """Off must be upstream: one job, whisper and silero tags, no attacher."""
    provider = WhisperStreamingProvider(
        BASE_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)

    session.handle_audio_chunk("c0", b"audio")

    assert mock_worker_pool.register_job.call_count == 1
    args, _ = mock_worker_pool.register_job.call_args
    assert args[0] == (WHISPER_TAG, SILERO_TAG)
    assert isinstance(args[2], WhisperStreamingProviderJob)
    assert args[3] == PROVIDER_KEY
    assert session._attacher is None
    assert session._diarization_job is None


def test_diarization_on_registers_a_second_job_fed_the_same_chunks(
    mock_logger, mock_worker_pool
):
    """
    The caption job is registered exactly as upstream does it; the
    diarization job goes to the pyannote tag with the suffixed label, on its
    own period, and every chunk reaches both.
    """
    provider = WhisperStreamingProvider(
        DIARIZED_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)

    session.handle_audio_chunk("c0", b"audio")

    assert mock_worker_pool.register_job.call_count == 2
    caption_args, _ = mock_worker_pool.register_job.call_args_list[0]
    diar_args, diar_kwargs = mock_worker_pool.register_job.call_args_list[1]
    assert caption_args[0] == (WHISPER_TAG, SILERO_TAG)
    assert caption_args[3] == PROVIDER_KEY
    assert diar_args[0] == (PYANNOTE_TAG,)
    assert diar_args[1] == 5000
    assert isinstance(diar_args[2], DiarizationJob)
    assert diar_args[3] == PROVIDER_KEY + DIARIZATION_JOB_LABEL_SUFFIX
    assert diar_kwargs["session_uid"] == "sess"
    # Caption admission is still decided by the caption worker only.
    assert session.admission_worker_id == 0

    session._job.queue_data.assert_called_once()
    (diar_batch,), _ = session._diarization_job.queue_data.call_args
    assert isinstance(diar_batch[0], DiarizationChunk)
    assert diar_batch[0].chunk_id == "c0"
    assert diar_batch[0].audio_bytes == b"audio"


def test_captions_are_emitted_immediately_with_known_labels_only(
    mock_logger, mock_worker_pool
):
    """
    A caption result is forwarded at once. Words diarization has covered
    carry labels, the rest None, and a finalized sequence gets an id.
    """
    provider = WhisperStreamingProvider(
        DIARIZED_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)
    session.handle_audio_chunk("c0", b"audio")
    results = []
    session.on(
        TranscriptionSessionInterface.TranscriptionResultEvent, results.append
    )

    _emit_result(
        session,
        session._diarization_job,
        DiarizationResult(
            segments=[SpeakerSegment(0.0, 4.0, "spk_0")],
            window_start=0.0,
            window_end=5.0,
            lag_sec=0.3,
            labels_minted=1,
        ),
    )
    _emit_result(
        session,
        session._job,
        TranscriptionResult(
            final=TranscriptionSequence(["a"], [0.0], [1.0]),
            in_progress=TranscriptionSequence(
                ["b", "c"], [2.0, 6.0], [3.0, 7.0]
            ),
        ),
    )

    assert len(results) == 1
    assert results[0].final.speakers == ["spk_0"]
    assert results[0].final.sequence_id == "s0"
    assert results[0].in_progress.speakers == ["spk_0", None]
    assert results[0].in_progress.sequence_id is None


def test_late_labels_arrive_as_speaker_label_events(
    mock_logger, mock_worker_pool
):
    """
    A finalized caption sent before its audio was diarized is followed by a
    SpeakerLabelsEvent naming its id once coverage reaches it. The caption
    itself was never held.
    """
    provider = WhisperStreamingProvider(
        DIARIZED_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)
    session.handle_audio_chunk("c0", b"audio")
    results = []
    updates = []
    session.on(
        TranscriptionSessionInterface.TranscriptionResultEvent, results.append
    )
    session.on(TranscriptionSessionInterface.SpeakerLabelsEvent, updates.append)

    _emit_result(
        session,
        session._job,
        TranscriptionResult(
            final=TranscriptionSequence(["a", "b"], [0.0, 1.0], [1.0, 2.0])
        ),
    )
    assert results[0].final.speakers == [None, None]
    assert not updates

    _emit_result(
        session,
        session._diarization_job,
        DiarizationResult(
            segments=[SpeakerSegment(0.0, 4.0, "spk_1")],
            window_start=0.0,
            window_end=5.0,
            lag_sec=0.3,
            labels_minted=1,
        ),
    )

    assert len(updates) == 1
    assert updates[0].sequence_id == results[0].final.sequence_id
    assert updates[0].speakers == ["spk_1", "spk_1"]
    assert updates[0].settled is True


def test_whisper_drops_are_fed_to_the_attacher(mock_logger, mock_worker_pool):
    """
    The caption job's buffer-full drop counter shifts its clock against the
    diarization job's; the session passes it on so labels stay aligned.
    """
    provider = WhisperStreamingProvider(
        DIARIZED_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)
    session.handle_audio_chunk("c0", b"audio")

    _emit_result(
        session,
        session._job,
        TranscriptionResult(
            in_progress=TranscriptionSequence(["a"], [0.0], [8.0])
        ),
    )
    _emit_result(
        session,
        session._job,
        TranscriptionResult(
            in_progress=TranscriptionSequence(["a"], [0.0], [9.0])
        ),
        counters={
            TranscriptionJobCounter.AUDIO_DROPPED_BUFFER_FULL_SECONDS: 2.0
        },
    )

    assert session._attacher.to_diarization_time(7.0) == 7.0
    assert session._attacher.to_diarization_time(9.0) == 11.0


def test_a_failed_diarization_job_does_not_end_the_session(
    mock_logger, mock_worker_pool
):
    """Labels stop; captions continue; no error reaches the client."""
    provider = WhisperStreamingProvider(
        DIARIZED_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)
    session.handle_audio_chunk("c0", b"audio")
    errors = []
    session.on(
        TranscriptionSessionInterface.TranscriptionErrorEvent, errors.append
    )

    callback = session._diarization_job.on.call_args[0][1]
    callback(JobException(RuntimeError("boom"), STATS))

    assert not errors
    assert session._diarization_job is None
    mock_logger.warning.assert_called()


def test_end_session_deregisters_both_jobs(mock_logger, mock_worker_pool):
    """Neither job may outlive the session."""
    provider = WhisperStreamingProvider(
        DIARIZED_CONFIG, mock_logger, mock_worker_pool, PROVIDER_KEY
    )
    session = provider.create_session("cfg", "sess", "room", mock_logger)
    session.handle_audio_chunk("c0", b"audio")
    caption_job, diarization_job = session._job, session._diarization_job

    session.end_session()

    caption_job.deregister.assert_called_once()
    diarization_job.deregister.assert_called_once()


def test_missing_diarization_context_fails_at_startup(mock_logger):
    """
    A diarization tag no worker owns is a configuration error at start-up,
    not a 1011 on the first audio chunk.
    """
    pool = MagicMock(spec=WorkerPool)
    pool.load_for_tags.return_value = []

    with pytest.raises(ValueError, match=PYANNOTE_TAG):
        WhisperStreamingProvider(
            DIARIZED_CONFIG, mock_logger, pool, PROVIDER_KEY
        )


def test_shared_worker_is_warned_about(mock_logger):
    """A worker running both captions and diarization is legal but warned."""
    pool = MagicMock(spec=WorkerPool)
    pool.load_for_tags.return_value = [_snapshot(0)]

    WhisperStreamingProvider(DIARIZED_CONFIG, mock_logger, pool, PROVIDER_KEY)

    mock_logger.warning.assert_called_once()
    assert "worker_ids of its own" in mock_logger.warning.call_args[0][0]


@pytest.mark.asyncio
async def test_health_is_down_when_the_diarization_context_is_missing(
    mock_logger,
):
    """Health names the missing diarization context the way it names whisper's."""
    pool = MagicMock(spec=WorkerPool)
    pool.load_for_tags.side_effect = lambda tags: (
        [_snapshot(0)] if tags == (WHISPER_TAG, SILERO_TAG) else []
    )
    pool.load_for_tags.return_value = None
    provider = WhisperStreamingProvider.__new__(WhisperStreamingProvider)
    provider._log = mock_logger
    provider.config = WhisperStreamingProvider(
        BASE_CONFIG, mock_logger, pool, PROVIDER_KEY
    ).config.model_copy(update={"diarization_detector": True})
    provider.worker_pool = pool
    provider.provider_key = PROVIDER_KEY
    provider._active_sessions = 0

    health = await provider.describe_health()

    assert health.status == ProviderStatus.DOWN
    assert health.model_loaded is False
    assert PYANNOTE_TAG in (health.detail or "")


def test_a_reconnect_within_the_grace_continues_the_speaker_memory(
    mock_logger, mock_worker_pool
):
    """
    Phase 2b: when a session's socket closes, the provider keeps its
    reconciler state and sequence counter in memory for the grace period;
    a new session with the same session_uid starts its diarization job from
    that state and continues the sequence ids. The memory is handed over
    once and is gone afterwards.
    """
    provider = WhisperStreamingProvider(
        {**DIARIZED_CONFIG, "diarization_reconnect_grace_sec": 60},
        mock_logger,
        mock_worker_pool,
        PROVIDER_KEY,
    )
    session = provider.create_session("cfg", "session-1", "room-1", mock_logger)
    session.handle_audio_chunk("c0", b"audio")
    state = SpeakerReconcilerState(next_label_id=2, version=3)
    _emit_result(
        session,
        session._diarization_job,
        DiarizationResult(
            segments=[SpeakerSegment(0.0, 4.0, "spk_1")],
            window_start=0.0,
            window_end=5.0,
            lag_sec=0.3,
            labels_minted=2,
            state=state,
        ),
    )
    session._attacher.continue_sequence_ids_from(5)
    session.end_session()
    assert provider.remembered_sessions == 1

    mock_worker_pool.register_job.reset_mock()
    rejoined = provider.create_session(
        "cfg", "session-1", "room-1", mock_logger
    )
    rejoined.handle_audio_chunk("c1", b"audio")

    assert provider.remembered_sessions == 0
    diarization_call = mock_worker_pool.register_job.call_args_list[1]
    job = diarization_call.args[2]
    assert isinstance(job, DiarizationJob)
    assert job.labels_minted == 2
    assert rejoined._attacher.next_sequence_id == 5
    # A third connection finds nothing: the memory was handed over
    assert provider.recall_speakers("session-1") is None


def test_speaker_memory_expires_after_the_grace_and_is_never_kept_without_uid(
    mock_logger, mock_worker_pool
):
    """
    Memory past its grace is dropped on the next sweep; a session without
    a session_uid, or a provider with the grace at 0, remembers nothing.
    """
    provider = WhisperStreamingProvider(
        {**DIARIZED_CONFIG, "diarization_reconnect_grace_sec": 0.01},
        mock_logger,
        mock_worker_pool,
        PROVIDER_KEY,
    )
    state = SpeakerReconcilerState(next_label_id=1)
    provider.remember_speakers("session-1", state, 1)
    provider.remember_speakers(None, state, 1)
    assert provider.remembered_sessions == 1
    time.sleep(0.02)
    assert provider.recall_speakers("session-1") is None
    assert provider.remembered_sessions == 0

    disabled = WhisperStreamingProvider(
        {**DIARIZED_CONFIG, "diarization_reconnect_grace_sec": 0},
        mock_logger,
        mock_worker_pool,
        PROVIDER_KEY,
    )
    disabled.remember_speakers("session-1", state, 1)
    assert disabled.remembered_sessions == 0
