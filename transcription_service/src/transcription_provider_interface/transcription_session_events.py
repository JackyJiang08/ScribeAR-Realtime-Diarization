"""
Defines the events a transcription session emits, shared by every emitter
that forwards them (the provider's session and the webserver's
per-connection stream service), so the declarations exist once
"""

from src.shared.utils.event_emitter import Event

from .speaker_label_update import SpeakerLabelUpdate
from .transcription_client_error import TranscriptionClientError
from .transcription_result import TranscriptionResult


class TranscriptionSessionEvents:
    """
    Mixin declaring the three session events. An emitter that forwards a
    session's output declares them by inheriting this class, so a consumer
    can subscribe with the same attribute names on the session and on the
    service that relays it.

    Events:
        TranscriptionResultEvent    - A caption result (final and/or
                                        in-progress sequences)
        TranscriptionErrorEvent     - A client-attributable error or an
                                        unexpected exception
        SpeakerLabelsEvent          - Speaker labels for a finalized sequence
                                        that was already emitted. Only a
                                        provider that runs diarization ever
                                        emits this; the others never do, and
                                        a consumer that ignores it still gets
                                        every caption
    """

    TranscriptionResultEvent = Event[TranscriptionResult](
        "TRANSCRIPTION_RESULT"
    )
    TranscriptionErrorEvent = Event[TranscriptionClientError | Exception](
        "TRANSCRIPTION_ERROR"
    )
    SpeakerLabelsEvent = Event[SpeakerLabelUpdate]("SPEAKER_LABELS")
