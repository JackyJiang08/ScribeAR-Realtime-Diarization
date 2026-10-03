"""
Defines SpeakerLabelAttacher, which attaches diarization labels to caption
words after the fact, without ever holding a caption back
"""

from dataclasses import dataclass, field

from src.shared.utils.speaker_reconciler import SpeakerSegment
from src.transcription_provider_interface import (
    SpeakerLabelUpdate,
    TranscriptionSequence,
)

# How much labelled timeline is kept behind the coverage watermark. Pending
# sequences are the only reader of old segments, and a sequence that has
# waited longer than the label timeout has been settled, so anything older
# than this is unreachable.
DEFAULT_HISTORY_SEC = 120.0


def assign_speaker(
    word_start: float, word_end: float, segments: list[SpeakerSegment]
) -> str | None:
    """
    The speaker whose segment overlaps the word the most, or None when no
    segment overlaps it at all

    Args:
        word_start  - Word start, same timeline as the segments
        word_end    - Word end
        segments    - Speaker segments to attribute against

    Returns:
        Speaker label, or None
    """
    best_speaker = None
    best_overlap = 0.0
    for segment in segments:
        overlap = min(word_end, segment.end) - max(word_start, segment.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.speaker
    return best_speaker


@dataclass
class _PendingSequence:
    """
    A finalized sequence that was sent before every word had a label
    """

    sequence_id: str
    starts: list[float]
    ends: list[float]
    speakers: list[str | None]
    # Index of the first word whose label has not been decided yet. Words are
    # time-ordered, so everything before it is frozen.
    undecided_from: int
    sent_at: float


@dataclass
class _DropEvent:
    """
    Audio Whisper dropped because its buffer was full, at the Whisper time it
    happened. Everything Whisper timestamps after this point sits that much
    earlier than the diarization job's timeline, which never drops audio.
    """

    whisper_time: float
    dropped_sec: float


@dataclass
class SpeakerLabelAttacher:
    """
    Attaches speaker labels to caption words from a separately produced,
    append-only diarization timeline

    Captions and diarization run in different worker processes and arrive at
    the session in any order. This class is the meeting point:

    - `add_coverage` takes each diarization result (reconciled, session-time
      speaker segments over the newest window) and extends a frozen timeline
      up to `window_end - edge_margin_sec`. Labels already on the timeline are
      never rewritten, so a word's label is decided exactly once, by the
      first diarization pass that covered it. A later pass can refine only
      the uncovered tail. Audio that no pass covered (the diarization job fell
      behind and skipped it) is decided as "no speaker".
    - `label_sequence` labels a caption sequence's words from that timeline
      at emission time. Words whose end lies beyond the watermark stay None;
      for a finalized sequence they are remembered and labelled later through
      a `SpeakerLabelUpdate`, which `add_coverage` returns.
    - A finalized sequence that waits longer than `label_timeout_sec` is
      settled with the labels it has, so a stalled diarization job can never
      leave a caption waiting forever.

    Timelines: Whisper's word timestamps count only the audio its buffer
    kept, while the diarization job counts every chunk it received. The two
    agree until Whisper drops audio because its buffer was full (a service
    stall), after which Whisper's clock lags by the dropped duration.
    `record_whisper_drop` keeps that offset so labels stay aligned.
    """

    edge_margin_sec: float = 0.5
    label_timeout_sec: float = 15.0
    history_sec: float = DEFAULT_HISTORY_SEC
    _segments: list[SpeakerSegment] = field(default_factory=list)
    _covered_through: float = 0.0
    _pending: dict[str, _PendingSequence] = field(default_factory=dict)
    _next_sequence_id: int = 0
    _drops: list[_DropEvent] = field(default_factory=list)
    _last_whisper_end: float = 0.0

    @property
    def covered_through(self) -> float:
        """
        Diarization-timeline instant up to which every word's label is decided
        """
        return self._covered_through

    @property
    def pending_sequences(self) -> int:
        """
        Finalized sequences still waiting for at least one label
        """
        return len(self._pending)

    def new_sequence_id(self) -> str:
        """
        Allocates the id a finalized sequence is sent with, unique per session
        """
        sequence_id = f"s{self._next_sequence_id}"
        self._next_sequence_id += 1
        return sequence_id

    def record_whisper_drop(self, dropped_sec: float) -> None:
        """
        Records audio Whisper dropped because its buffer was full

        Args:
            dropped_sec - Seconds dropped, from the job's
                            `audio_dropped_buffer_full_seconds` counter

        The drop is placed at the newest Whisper time seen so far. The exact
        chunk is not reported, so words emitted between the drop and the next
        result may be a few seconds off; drops only happen when the caption
        worker has stalled for longer than its buffer holds, and the
        placement is exact for everything after the next result.
        """
        if dropped_sec <= 0:
            return
        self._drops.append(_DropEvent(self._last_whisper_end, dropped_sec))

    def note_whisper_end(self, end_time: float) -> None:
        """
        Remembers the newest Whisper timestamp seen, where a later drop is
        placed

        Args:
            end_time    - Newest word end in a Whisper result
        """
        self._last_whisper_end = max(self._last_whisper_end, end_time)

    def to_diarization_time(self, whisper_time: float) -> float:
        """
        Maps a Whisper timestamp onto the diarization job's timeline

        Args:
            whisper_time    - Timestamp as Whisper reported it

        Returns:
            The same instant on the diarization timeline
        """
        offset = 0.0
        for drop in self._drops:
            if drop.whisper_time <= whisper_time:
                offset += drop.dropped_sec
        return whisper_time + offset

    def add_coverage(
        self,
        segments: list[SpeakerSegment],
        window_start: float,
        window_end: float,
        now: float,
    ) -> list[SpeakerLabelUpdate]:
        """
        Extends the frozen timeline with one diarization pass and labels the
        pending sequences it reaches

        Args:
            segments        - Reconciled segments over [window_start,
                                window_end], diarization timeline
            window_start    - Start of the diarized window
            window_end      - End of the diarized window
            now             - Wall clock, for the label timeout

        Returns:
            Updates for pending sequences that gained labels or were settled
        """
        new_through = max(
            self._covered_through, window_end - self.edge_margin_sec
        )
        if new_through > self._covered_through:
            lower = max(self._covered_through, window_start)
            for segment in segments:
                start = max(segment.start, lower)
                end = min(segment.end, new_through)
                if end > start:
                    self._segments.append(
                        SpeakerSegment(start, end, segment.speaker)
                    )
            self._covered_through = new_through
            self._trim_history()
        return self._flush_pending(now)

    def label_sequence(
        self, sequence: TranscriptionSequence, final: bool, now: float
    ) -> None:
        """
        Labels a caption sequence's words from the frozen timeline, in place

        Args:
            sequence    - Sequence about to be emitted; `speakers` is set and,
                            for a finalized sequence, `sequence_id`
            final       - Whether this is a finalized sequence (append-only
                            on the client, so late labels need an id) or the
                            in-progress tail (replaced every tick, so late
                            labels ride on the next tick instead)
            now         - Wall clock, for the label timeout
        """
        if sequence.starts is None or sequence.ends is None:
            sequence.speakers = [None] * len(sequence.text)
            if final:
                sequence.sequence_id = self.new_sequence_id()
            return

        self.note_whisper_end(max(sequence.ends, default=0.0))
        starts = [self.to_diarization_time(t) for t in sequence.starts]
        ends = [self.to_diarization_time(t) for t in sequence.ends]
        speakers, undecided_from = self._label_words(starts, ends, 0)
        sequence.speakers = speakers
        if not final:
            return

        sequence.sequence_id = self.new_sequence_id()
        if undecided_from < len(speakers):
            self._pending[sequence.sequence_id] = _PendingSequence(
                sequence_id=sequence.sequence_id,
                starts=starts,
                ends=ends,
                speakers=list(speakers),
                undecided_from=undecided_from,
                sent_at=now,
            )

    def expire(self, now: float) -> list[SpeakerLabelUpdate]:
        """
        Settles pending sequences that have waited past the label timeout

        Args:
            now - Wall clock

        Returns:
            One settling update per timed-out sequence
        """
        updates: list[SpeakerLabelUpdate] = []
        for sequence_id in list(self._pending):
            pending = self._pending[sequence_id]
            if now - pending.sent_at < self.label_timeout_sec:
                continue
            del self._pending[sequence_id]
            updates.append(
                SpeakerLabelUpdate(sequence_id, list(pending.speakers), True)
            )
        return updates

    def _label_words(
        self, starts: list[float], ends: list[float], from_index: int
    ) -> tuple[list[str | None], int]:
        """
        Labels words from `from_index` on while their end is covered

        Returns:
            (labels for every word, index of the first undecided word)
        """
        speakers: list[str | None] = [None] * len(starts)
        undecided_from = len(starts)
        for index in range(from_index, len(starts)):
            if ends[index] > self._covered_through:
                undecided_from = index
                break
            speakers[index] = assign_speaker(
                starts[index], ends[index], self._segments
            )
        return speakers, undecided_from

    def _flush_pending(self, now: float) -> list[SpeakerLabelUpdate]:
        updates: list[SpeakerLabelUpdate] = []
        for sequence_id in list(self._pending):
            pending = self._pending[sequence_id]
            labels, undecided_from = self._label_words(
                pending.starts, pending.ends, pending.undecided_from
            )
            if undecided_from == pending.undecided_from:
                continue
            for index in range(pending.undecided_from, undecided_from):
                pending.speakers[index] = labels[index]
            pending.undecided_from = undecided_from
            settled = undecided_from >= len(pending.speakers)
            if settled:
                del self._pending[sequence_id]
            updates.append(
                SpeakerLabelUpdate(sequence_id, list(pending.speakers), settled)
            )
        updates.extend(self.expire(now))
        return updates

    def _trim_history(self) -> None:
        horizon = self._covered_through - self.history_sec
        if self._segments and self._segments[0].end < horizon:
            self._segments = [s for s in self._segments if s.end >= horizon]
