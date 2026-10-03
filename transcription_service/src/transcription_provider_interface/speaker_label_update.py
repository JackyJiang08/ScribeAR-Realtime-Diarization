"""
Defines SpeakerLabelUpdate, the late-arriving speaker labels for a caption
sequence that was already shown
"""

from dataclasses import dataclass


@dataclass
class SpeakerLabelUpdate:
    """
    Speaker labels for a finalized transcription sequence that was emitted
    before diarization had covered its audio.

    Captions are never held back for speaker labels. A finalized sequence is
    sent as soon as Whisper finalizes it, carrying a `sequence_id` and
    whatever labels are already known; this message attaches the rest once
    the diarization job's coverage reaches the sequence's words. A client that
    ignores the message still shows the caption text.

    Properties:
        sequence_id - Id the finalized `TranscriptionSequence` was sent with
        speakers    - Labels aligned with that sequence's `text`; None for a
                        word that is still unlabelled, or that will never be
                        labelled once `settled` is True
        settled     - True when no further update will follow for this
                        sequence: every word is labelled, or the ones that
                        are not never will be (the audio was never diarized)
    """

    sequence_id: str
    speakers: list[str | None]
    settled: bool
