"""
Unit tests for folding fragments in SpeakerReconciler: a label minted by
one pass whose audio the next, overlapping pass re-labels as another
speaker is folded into that speaker before its captions settle
"""

import numpy as np

from src.shared.utils.speaker_reconciler import (
    SpeakerReconciler,
    SpeakerReconcilerConfig,
    SpeakerSegment,
)


def _voice(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)


def _config(**overrides) -> SpeakerReconcilerConfig:
    defaults = {
        "match_threshold": 0.9,
        "new_speaker_threshold": 0.5,
        "attach_threshold": 0.2,
        "min_mint_duration_sec": 1.0,
        "overlap_bonus": 0.0,
        "merge_threshold": 1.0,
        "fragment_fold_sec": 7.5,
        "fragment_fold_fraction": 0.8,
    }
    defaults.update(overrides)
    return SpeakerReconcilerConfig(**defaults)


def _split_then_rejoin(reconciler):
    """
    Pass 1 (0 to 10 s) hears speaker A at 0 degrees for 0 to 6 s and a
    second voice at 90 degrees for 6 to 10 s, which mints spk_1. Pass 2
    (5 to 15 s) hears all of 5 to 15 s as A: the second voice was a split
    of A. Returns pass 2's output
    """
    reconciler.reconcile(
        [SpeakerSegment(0.0, 6.0, "A"), SpeakerSegment(6.0, 10.0, "B")],
        {"A": _voice(0), "B": _voice(90)},
    )
    assert reconciler.labels_minted == 2
    return reconciler.reconcile(
        [SpeakerSegment(5.0, 15.0, "A")], {"A": _voice(0)}
    )


def test_a_fragment_the_next_pass_relabels_is_folded():
    """
    Test the fragment label folds into the speaker whose label the next
    pass puts on the same audio: the fold is reported as a merge, the
    previous run is relabelled so the attacher can follow, and the
    fragment's speaker is gone from memory
    """
    reconciler = SpeakerReconciler(config=_config())
    out = _split_then_rejoin(reconciler)

    assert [s.speaker for s in out] == ["spk_0"]
    assert reconciler.last_merges == [("spk_1", "spk_0")]
    assert [s.label for s in reconciler.speakers] == ["spk_0"]
    # The previous run's view of 6 to 10 s now carries the senior label
    assert {s.speaker for s in reconciler._state.previous} == {"spk_0"}


def test_a_voice_the_next_pass_hears_again_is_kept():
    """
    Test a new voice confirmed by the next pass (heard again under its own
    label on the overlapping audio) keeps its label
    """
    reconciler = SpeakerReconciler(config=_config())
    reconciler.reconcile(
        [SpeakerSegment(0.0, 6.0, "A"), SpeakerSegment(6.0, 10.0, "B")],
        {"A": _voice(0), "B": _voice(90)},
    )
    out = reconciler.reconcile(
        [SpeakerSegment(5.0, 6.0, "A"), SpeakerSegment(6.0, 15.0, "B")],
        {"A": _voice(0), "B": _voice(90)},
    )
    assert [s.speaker for s in out] == ["spk_0", "spk_1"]
    assert reconciler.last_merges == []
    assert reconciler.labels_minted == 2


def test_a_fragment_outside_the_next_window_is_left_alone():
    """
    Test a label whose audio the next window does not reach is neither
    folded nor confirmed: no evidence, no change
    """
    reconciler = SpeakerReconciler(config=_config())
    reconciler.reconcile(
        [SpeakerSegment(0.0, 3.0, "B"), SpeakerSegment(3.0, 10.0, "A")],
        {"A": _voice(0), "B": _voice(90)},
    )
    out = reconciler.reconcile(
        [SpeakerSegment(5.0, 15.0, "A")], {"A": _voice(0)}
    )
    assert [s.speaker for s in out] == ["spk_1"]
    assert reconciler.last_merges == []
    assert len(reconciler.speakers) == 2


def test_fold_needs_enough_of_the_fragment_covered():
    """
    Test the fold requires `fragment_fold_fraction` of the fragment's audio
    inside the window to be re-labelled: with only 1 s of 4 s covered by
    another speaker the fragment stays
    """
    reconciler = SpeakerReconciler(config=_config())
    reconciler.reconcile(
        [SpeakerSegment(0.0, 6.0, "A"), SpeakerSegment(6.0, 10.0, "B")],
        {"A": _voice(0), "B": _voice(90)},
    )
    out = reconciler.reconcile(
        [SpeakerSegment(5.0, 7.0, "A"), SpeakerSegment(10.0, 15.0, "A")],
        {"A": _voice(0)},
    )
    assert [s.speaker for s in out] == ["spk_0", "spk_0"]
    assert reconciler.last_merges == []
    assert len(reconciler.speakers) == 2


def test_an_older_label_is_never_folded():
    """
    Test the fold only applies within `fragment_fold_sec` of minting: a
    label two passes old is established and keeps its audio
    """
    reconciler = SpeakerReconciler(config=_config(fragment_fold_sec=4.0))
    out = _split_then_rejoin(reconciler)
    # The fragment is 5 s old at the second pass, past the 4 s limit
    assert [s.speaker for s in out] == ["spk_0"]
    assert reconciler.last_merges == []
    assert len(reconciler.speakers) == 2


def test_fold_is_disabled_at_zero():
    """
    Test `fragment_fold_sec` 0 keeps every minted label
    """
    reconciler = SpeakerReconciler(config=_config(fragment_fold_sec=0.0))
    _split_then_rejoin(reconciler)
    assert reconciler.last_merges == []
    assert len(reconciler.speakers) == 2
