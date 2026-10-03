"""
Public exports for speaker attribution
"""

from .speaker_label_attacher import (
    DEFAULT_CONFIDENCE,
    DEFAULT_HISTORY_SEC,
    LabelledSpan,
    SpeakerLabelAttacher,
    assign_speaker,
)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_HISTORY_SEC",
    "LabelledSpan",
    "SpeakerLabelAttacher",
    "assign_speaker",
]
