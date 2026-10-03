"""
Defines messages for transcription stream websocket server sent messages
"""

# The wire-format sequence deliberately mirrors the provider interface's
# TranscriptionSequence field for field; the two are kept apart so the wire
# format can only change on purpose.
# pylint: disable=duplicate-code

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from src.webserver.shared.json_server_message import JsonServerMessage


class ServerMessageTypes(StrEnum):
    """
    Defines possible server send message types
    """

    TRANSCRIPT = "transcript"
    # Speaker labels for a finalized sequence that was already sent. Emitted
    # only when diarization is on; a client that does not know the type can
    # ignore it and still has every caption.
    SPEAKERS_UPDATE = "speakers_update"


@dataclass
class TranscriptSequence:
    """
    A transcription sequence with text and optional timestamps
    """

    text: list[str]
    starts: list[float] | None = None
    ends: list[float] | None = None
    speakers: list[str | None] | None = None
    # Set on finalized sequences when diarization is on, so a later
    # `speakers_update` can name the sequence it labels. Null otherwise.
    sequence_id: str | None = None


@dataclass
class SpeakersUpdateMessage(JsonServerMessage):
    """
    Late speaker labels for a finalized transcript sequence

    `speakers` is aligned with the sequence's `text`; a null entry is a word
    without a label yet, or - once `settled` is true - a word that will never
    get one. A label already sent for a word is never changed by a later
    update; updates only fill in nulls.
    """

    sequence_id: str
    speakers: list[str | None]
    settled: bool
    type: Literal[ServerMessageTypes.SPEAKERS_UPDATE] = (
        ServerMessageTypes.SPEAKERS_UPDATE
    )


@dataclass
class TranscriptMessage(JsonServerMessage):
    """
    Message containing both finalized and in-progress transcription data
    """

    final: TranscriptSequence | None = None
    in_progress: TranscriptSequence | None = None
    # Ids of the source audio chunks that contributed to each transcript,
    # echoed back so the node server can measure end-to-end latency.
    final_chunk_ids: list[str] | None = None
    in_progress_chunk_ids: list[str] | None = None
    type: Literal[ServerMessageTypes.TRANSCRIPT] = ServerMessageTypes.TRANSCRIPT
