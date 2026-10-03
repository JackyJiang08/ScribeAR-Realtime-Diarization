"""
Unit tests for SpeakerLabelAttacher: captions get labels from whatever
diarization has covered, late labels arrive as updates, and a label that was
sent is never changed.
"""

from src.shared.utils.speaker_attribution import (
    SpeakerLabelAttacher,
    assign_speaker,
)
from src.shared.utils.speaker_reconciler import SpeakerSegment
from src.transcription_provider_interface import TranscriptionSequence


def _sequence(words: list[tuple[str, float, float]]) -> TranscriptionSequence:
    """A sequence from (text, start, end) tuples."""
    return TranscriptionSequence(
        text=[w[0] for w in words],
        starts=[w[1] for w in words],
        ends=[w[2] for w in words],
    )


def test_assign_speaker_picks_the_largest_overlap():
    """A word straddling two speakers gets the one it overlaps the most."""
    segments = [
        SpeakerSegment(0.0, 1.0, "spk_0"),
        SpeakerSegment(1.0, 3.0, "spk_1"),
    ]

    assert assign_speaker(0.8, 1.5, segments) == "spk_1"
    assert assign_speaker(0.2, 1.1, segments) == "spk_0"
    assert assign_speaker(5.0, 6.0, segments) is None


def test_in_progress_words_are_labelled_from_coverage_without_waiting():
    """
    Words already covered get their label at emission; words past the
    coverage watermark stay None and nothing is remembered for them (the
    in-progress tail is replaced every tick anyway).
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5)
    attacher.add_coverage(
        [SpeakerSegment(0.0, 4.0, "spk_0")], 0.0, 5.0, now=100.0
    )
    sequence = _sequence([("a", 0.0, 1.0), ("b", 3.0, 4.0), ("c", 6.0, 7.0)])

    attacher.label_sequence(sequence, final=False, now=100.0)

    assert sequence.speakers == ["spk_0", "spk_0", None]
    assert sequence.sequence_id is None
    assert attacher.pending_sequences == 0


def test_final_sequence_gets_an_id_and_its_late_labels_arrive_as_an_update():
    """
    A finalized sequence emitted before diarization covered it is sent with
    None labels and an id; the next coverage that reaches its words produces
    an update naming that id.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5)
    sequence = _sequence([("a", 0.0, 1.0), ("b", 1.0, 2.0)])

    attacher.label_sequence(sequence, final=True, now=100.0)

    assert sequence.sequence_id == "s0"
    assert sequence.speakers == [None, None]
    assert attacher.pending_sequences == 1

    updates = attacher.add_coverage(
        [SpeakerSegment(0.0, 2.5, "spk_1")], 0.0, 5.0, now=101.0
    )

    assert len(updates) == 1
    assert updates[0].sequence_id == "s0"
    assert updates[0].speakers == ["spk_1", "spk_1"]
    assert updates[0].settled is True
    assert attacher.pending_sequences == 0


def test_partial_coverage_sends_a_partial_update_then_settles():
    """
    Coverage that reaches only the first words sends those labels straight
    away (unsettled); the rest follow when their audio is covered.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5)
    sequence = _sequence([("a", 0.0, 1.0), ("b", 6.0, 7.0)])
    attacher.label_sequence(sequence, final=True, now=100.0)

    first = attacher.add_coverage(
        [SpeakerSegment(0.0, 4.0, "spk_0")], 0.0, 5.0, now=101.0
    )
    second = attacher.add_coverage(
        [SpeakerSegment(5.0, 9.0, "spk_1")], 5.0, 10.0, now=106.0
    )

    assert [u.speakers for u in first] == [["spk_0", None]]
    assert first[0].settled is False
    assert [u.speakers for u in second] == [["spk_0", "spk_1"]]
    assert second[0].settled is True


def test_labels_already_on_the_timeline_are_never_rewritten():
    """
    A later pass that relabels audio an earlier pass already covered changes
    nothing: the first label a region got is the one every word keeps.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5)
    attacher.add_coverage(
        [SpeakerSegment(0.0, 4.5, "spk_0")], 0.0, 5.0, now=100.0
    )
    attacher.add_coverage(
        [SpeakerSegment(0.0, 9.5, "spk_1")], 0.0, 10.0, now=105.0
    )
    sequence = _sequence([("a", 1.0, 2.0), ("b", 6.0, 7.0)])

    attacher.label_sequence(sequence, final=False, now=105.0)

    assert sequence.speakers == ["spk_0", "spk_1"]


def test_the_edge_of_a_window_is_left_for_the_next_pass():
    """
    The last `edge_margin_sec` of a window is not decided by that window, so
    a word ending there waits for the next pass rather than taking a label
    from the segmentation's least-informed stretch.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=1.0)
    attacher.add_coverage(
        [SpeakerSegment(0.0, 5.0, "spk_0")], 0.0, 5.0, now=100.0
    )
    sequence = _sequence([("a", 3.0, 3.9), ("b", 4.0, 4.5)])

    attacher.label_sequence(sequence, final=False, now=100.0)

    assert attacher.covered_through == 4.0
    assert sequence.speakers == ["spk_0", None]


def test_audio_no_pass_covered_settles_as_unlabelled():
    """
    When the diarization job skipped ahead (window_start beyond the
    watermark), the gap is decided as "no speaker" so captions there settle
    instead of waiting forever.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5)
    sequence = _sequence([("a", 6.0, 7.0), ("b", 21.0, 22.0)])
    attacher.label_sequence(sequence, final=True, now=100.0)

    updates = attacher.add_coverage(
        [SpeakerSegment(20.0, 29.0, "spk_0")], 20.0, 30.0, now=101.0
    )

    assert len(updates) == 1
    assert updates[0].speakers == [None, "spk_0"]
    assert updates[0].settled is True


def test_a_pending_sequence_times_out_with_the_labels_it_has():
    """
    A stalled diarization job must not leave a caption pending forever:
    past the timeout the sequence is settled with whatever it has.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5, label_timeout_sec=10)
    sequence = _sequence([("a", 0.0, 1.0)])
    attacher.label_sequence(sequence, final=True, now=100.0)

    assert not attacher.expire(now=105.0)
    updates = attacher.expire(now=111.0)

    assert len(updates) == 1
    assert updates[0].sequence_id == "s0"
    assert updates[0].speakers == [None]
    assert updates[0].settled is True
    assert attacher.pending_sequences == 0


def test_whisper_drops_shift_later_words_onto_the_diarization_timeline():
    """
    After Whisper drops audio its timestamps lag the diarization job's clock
    by the dropped duration; a word at Whisper time 10 s is diarization time
    13 s once 3 s were dropped at Whisper time 8 s.
    """
    attacher = SpeakerLabelAttacher(edge_margin_sec=0.5)
    attacher.add_coverage(
        [
            SpeakerSegment(0.0, 9.0, "spk_0"),
            SpeakerSegment(12.0, 19.0, "spk_1"),
        ],
        0.0,
        20.0,
        now=100.0,
    )
    attacher.note_whisper_end(8.0)
    attacher.record_whisper_drop(3.0)
    sequence = _sequence([("a", 5.0, 6.0), ("b", 10.0, 11.0)])

    attacher.label_sequence(sequence, final=False, now=100.0)

    assert attacher.to_diarization_time(10.0) == 13.0
    assert sequence.speakers == ["spk_0", "spk_1"]


def test_sequence_without_timestamps_is_labelled_none_and_not_pending():
    """A sequence with no timing cannot be attributed and never waits."""
    attacher = SpeakerLabelAttacher()
    sequence = TranscriptionSequence(text=["a", "b"])

    attacher.label_sequence(sequence, final=True, now=100.0)

    assert sequence.speakers == [None, None]
    assert sequence.sequence_id == "s0"
    assert attacher.pending_sequences == 0
