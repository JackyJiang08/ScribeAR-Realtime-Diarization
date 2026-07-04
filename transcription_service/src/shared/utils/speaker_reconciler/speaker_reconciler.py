"""
Defines SpeakerReconciler for keeping speaker labels stable across
repeated diarization runs
"""

from dataclasses import dataclass, replace


@dataclass
class SpeakerSegment:
    """
    Speaker label active over a time range

    Properties:
        start   - Segment start in seconds relative to transcription session
        end     - Segment end in seconds relative to transcription session
        speaker - Speaker label for this time range
    """

    start: float
    end: float
    speaker: str


class SpeakerReconciler:
    """
    Maps per-run diarization labels to stable session-wide labels.

    Diarization pipelines assign arbitrary labels per run (e.g. SPEAKER_00),
    so the same physical speaker can receive a different label every time the
    rolling audio buffer is re-diarized. Because consecutive runs overlap in
    time, a run's labels can be matched to the previous run's session-wide
    labels by voting on overlap duration: each raw label is mapped to the
    session label it overlaps with the most, one-to-one, and raw labels with
    no overlapping predecessor mint a fresh session label.
    """

    def __init__(self, label_prefix: str = "spk_"):
        self._label_prefix = label_prefix
        self._previous: list[SpeakerSegment] = []
        self._next_label_id = 0

    def reconcile(self, segments: list[SpeakerSegment]) -> list[SpeakerSegment]:
        """
        Convert one diarization run's raw labels to session-wide labels

        Args:
            segments    - Segments from one diarization run, with timestamps
                            relative to the transcription session

        Returns:
            Segments with raw labels replaced by stable session-wide labels
        """
        # Keep the last non-empty run so continuity survives silent ticks
        if len(segments) == 0:
            return []

        votes: dict[tuple[str, str], float] = {}
        for segment in segments:
            for previous in self._previous:
                overlap = min(segment.end, previous.end) - max(
                    segment.start, previous.start
                )
                if overlap <= 0:
                    continue
                pair = (segment.speaker, previous.speaker)
                votes[pair] = votes.get(pair, 0.0) + overlap

        mapping: dict[str, str] = {}
        claimed: set[str] = set()
        for (raw, session_label), _ in sorted(
            votes.items(), key=lambda item: item[1], reverse=True
        ):
            if raw in mapping or session_label in claimed:
                continue
            mapping[raw] = session_label
            claimed.add(session_label)

        for segment in segments:
            if segment.speaker not in mapping:
                mapping[segment.speaker] = (
                    f"{self._label_prefix}{self._next_label_id}"
                )
                self._next_label_id += 1

        reconciled = [
            replace(segment, speaker=mapping[segment.speaker])
            for segment in segments
        ]
        self._previous = reconciled
        return reconciled
