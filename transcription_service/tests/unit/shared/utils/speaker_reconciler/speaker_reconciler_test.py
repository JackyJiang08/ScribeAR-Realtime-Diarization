"""
Unit tests for SpeakerReconciler
"""

from src.shared.utils.speaker_reconciler import (
    SpeakerReconciler,
    SpeakerSegment,
)


def test_first_run_mints_sequential_session_labels():
    """
    Test first run assigns spk_0, spk_1... in order of appearance
    """
    reconciler = SpeakerReconciler()

    result = reconciler.reconcile(
        [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(2.0, 4.0, "SPEAKER_01"),
        ]
    )

    assert [segment.speaker for segment in result] == ["spk_0", "spk_1"]


def test_swapped_raw_labels_keep_stable_session_labels():
    """
    Test raw label flips between runs map back to stable session labels
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 2.0, "SPEAKER_00"),
            SpeakerSegment(2.0, 4.0, "SPEAKER_01"),
        ]
    )

    # Second run over an overlapping window where the diarizer flipped labels
    result = reconciler.reconcile(
        [
            SpeakerSegment(0.0, 2.0, "SPEAKER_01"),
            SpeakerSegment(2.0, 5.0, "SPEAKER_00"),
        ]
    )

    assert [segment.speaker for segment in result] == ["spk_0", "spk_1"]


def test_new_speaker_gets_fresh_session_label():
    """
    Test a raw label with no history mints the next session label
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile([SpeakerSegment(0.0, 3.0, "SPEAKER_00")])

    result = reconciler.reconcile(
        [
            SpeakerSegment(0.0, 3.0, "SPEAKER_00"),
            SpeakerSegment(3.0, 5.0, "SPEAKER_01"),
        ]
    )

    assert [segment.speaker for segment in result] == ["spk_0", "spk_1"]


def test_mapping_prefers_largest_overlap():
    """
    Test a raw label maps to the session label it overlaps the most
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 1.0, "SPEAKER_00"),
            SpeakerSegment(1.0, 5.0, "SPEAKER_01"),
        ]
    )

    # A single raw label spanning both previous speakers maps to the one
    # it overlaps the most (spk_1 with 4s vs spk_0 with 1s)
    result = reconciler.reconcile([SpeakerSegment(0.0, 5.0, "SPEAKER_00")])

    assert [segment.speaker for segment in result] == ["spk_1"]


def test_empty_run_returns_empty_and_keeps_state():
    """
    Test a silent run returns empty without erasing label continuity
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile([SpeakerSegment(0.0, 2.0, "SPEAKER_00")])
    assert reconciler.reconcile([]) == []

    # A silent tick must not erase continuity: the next overlapping run
    # still maps onto the previously established session label
    result = reconciler.reconcile([SpeakerSegment(0.0, 2.0, "SPEAKER_01")])
    assert [segment.speaker for segment in result] == ["spk_0"]
