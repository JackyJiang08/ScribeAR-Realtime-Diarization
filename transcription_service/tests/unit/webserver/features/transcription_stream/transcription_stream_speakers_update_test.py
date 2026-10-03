"""
Unit tests for the speakers_update server message: late speaker labels a
session emits are serialized onto the socket, and the transcript message
carries a sequence id only when the provider set one.
"""

# pylint: disable=protected-access

import json
from unittest.mock import MagicMock

import pytest

from src.transcription_provider_interface import (
    SpeakerLabelUpdate,
    TranscriptionResult,
    TranscriptionSequence,
    TranscriptionSessionInterface,
)
from src.webserver.features.transcription_stream import (
    TranscriptionStreamController,
)
from src.webserver.features.transcription_stream.transcription_stream_messages import (
    SpeakersUpdateMessage,
    TranscriptMessage,
    TranscriptSequence,
)

from .conftest import (
    VALID_AUTH_MESSAGE,
    VALID_CONFIG_MESSAGE,
    MockTranscriptionSession,
)


@pytest.mark.asyncio
async def test_controller_sends_speaker_label_updates(
    controller: TranscriptionStreamController,
    mock_auth_service: MagicMock,
    mock_provider_registry: MagicMock,
    mock_send_method: MagicMock,
):
    """A SpeakerLabelsEvent becomes a speakers_update message."""
    mock_session = MockTranscriptionSession()
    mock_auth_service.is_authenticated.return_value = True
    mock_provider_registry.create_session.return_value = mock_session
    await controller._handle_text_message(VALID_AUTH_MESSAGE)
    await controller._handle_text_message(VALID_CONFIG_MESSAGE)

    mock_session.emit(
        TranscriptionSessionInterface.SpeakerLabelsEvent,
        SpeakerLabelUpdate("s3", ["spk_0", None], False),
    )

    mock_send_method.assert_called_once_with(
        SpeakersUpdateMessage(
            sequence_id="s3", speakers=["spk_0", None], settled=False
        )
    )


@pytest.mark.asyncio
async def test_transcript_message_carries_the_sequence_id_and_labels(
    controller: TranscriptionStreamController,
    mock_auth_service: MagicMock,
    mock_provider_registry: MagicMock,
    mock_send_method: MagicMock,
):
    """
    A finalized sequence with a sequence id (diarization on) is sent with it
    and its known labels; without one (diarization off) the fields are null,
    exactly as before.
    """
    mock_session = MockTranscriptionSession()
    mock_auth_service.is_authenticated.return_value = True
    mock_provider_registry.create_session.return_value = mock_session
    await controller._handle_text_message(VALID_AUTH_MESSAGE)
    await controller._handle_text_message(VALID_CONFIG_MESSAGE)

    mock_session.emit(
        TranscriptionSessionInterface.TranscriptionResultEvent,
        TranscriptionResult(
            final=TranscriptionSequence(
                ["a"], [0.0], [1.0], speakers=["spk_0"], sequence_id="s1"
            )
        ),
    )

    mock_send_method.assert_called_once_with(
        TranscriptMessage(
            final=TranscriptSequence(
                ["a"], [0.0], [1.0], speakers=["spk_0"], sequence_id="s1"
            ),
            in_progress=None,
        )
    )


def test_speakers_update_message_serializes_to_the_documented_shape():
    """The wire shape the TypeScript schemas mirror."""
    payload = json.loads(
        SpeakersUpdateMessage("s9", ["spk_1", None], True).serialize()
    )

    assert payload == {
        "sequence_id": "s9",
        "speakers": ["spk_1", None],
        "settled": True,
        "type": "speakers_update",
    }


def test_transcript_message_without_diarization_is_unchanged_on_the_wire():
    """Diarization off: speakers and sequence_id are null, nothing else."""
    payload = json.loads(
        TranscriptMessage(
            final=TranscriptSequence(["a"], [0.0], [1.0]), in_progress=None
        ).serialize()
    )

    assert payload["final"] == {
        "text": ["a"],
        "starts": [0.0],
        "ends": [1.0],
        "speakers": None,
        "sequence_id": None,
    }
