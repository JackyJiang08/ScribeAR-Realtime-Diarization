"""
Unit tests for merging converged speakers in SpeakerReconciler: a new
label whose centroid converges to an existing speaker folds back into it,
and the age limit keeps established speakers apart
"""

import numpy as np

from src.shared.utils.speaker_reconciler import (
    SpeakerReconciler,
    SpeakerReconcilerConfig,
    SpeakerSegment,
)


def _voice(angle_deg: float) -> np.ndarray:
    """
    A unit embedding in the plane; the cosine between two voices is the
    cosine of their angle, so similarity is easy to place
    """
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)


def _run(reconciler, start, end, raw, voice):
    return reconciler.reconcile([SpeakerSegment(start, end, raw)], {raw: voice})


def _config(**overrides) -> SpeakerReconcilerConfig:
    # A short centroid memory lets a centroid follow the newest clusters, so
    # the drift of a new label towards an established one is quick to stage
    defaults = {
        "match_threshold": 0.9,
        "new_speaker_threshold": 0.5,
        "attach_threshold": 0.2,
        "min_mint_duration_sec": 1.0,
        "overlap_bonus": 0.0,
        "centroid_memory_sec": 1.0,
        "merge_threshold": 0.8,
        "merge_max_age_sec": 20.0,
    }
    defaults.update(overrides)
    return SpeakerReconcilerConfig(**defaults)


def _drift_second_voice_towards_first(reconciler):
    """
    spk_0 at 0 degrees, spk_1 minted at 70 degrees (10 s), then spk_1 is
    heard at 60, 45 and 30 degrees: its centroid converges on spk_0 and the
    last run ends at 25 s
    """
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    _run(reconciler, 5.0, 10.0, "B", _voice(70))
    assert reconciler.labels_minted == 2
    _run(reconciler, 10.0, 15.0, "B", _voice(60))
    _run(reconciler, 15.0, 20.0, "B", _voice(45))
    return _run(reconciler, 20.0, 25.0, "B", _voice(30))


def test_a_new_label_that_converges_is_merged_into_the_senior_speaker():
    """
    Test the junior label folds into the senior one as soon as their
    centroids reach the merge threshold: the run's segments carry the
    senior label, the merge is reported, and the memory holds one speaker
    """
    reconciler = SpeakerReconciler(config=_config())
    last = _drift_second_voice_towards_first(reconciler)

    assert [s.speaker for s in last] == ["spk_0"]
    assert reconciler.last_merges == [("spk_1", "spk_0")]
    assert [s.label for s in reconciler.speakers] == ["spk_0"]
    assert reconciler.last_mapping == {"B": "spk_0"}
    assert "spk_1" not in reconciler.last_confidence


def test_an_established_label_is_never_merged_under_the_age_limit():
    """
    Test the same convergence does not merge when the junior label is
    older than `merge_max_age_sec`: two established speakers stay apart
    however alike their centroids become
    """
    reconciler = SpeakerReconciler(config=_config(merge_max_age_sec=10.0))
    last = _drift_second_voice_towards_first(reconciler)

    # spk_1 was minted at 10 s and the run ends at 25 s: 15 s old
    assert [s.speaker for s in last] == ["spk_1"]
    assert not reconciler.last_merges
    assert [s.label for s in reconciler.speakers] == ["spk_0", "spk_1"]


def test_no_age_limit_merges_regardless_of_age():
    """
    Test `merge_max_age_sec` 0 keeps the earlier behaviour: any two
    speakers whose centroids converge are merged
    """
    reconciler = SpeakerReconciler(config=_config(merge_max_age_sec=0.0))
    last = _drift_second_voice_towards_first(reconciler)

    assert [s.speaker for s in last] == ["spk_0"]
    assert reconciler.last_merges == [("spk_1", "spk_0")]


def test_merging_is_off_by_default():
    """
    Test the default configuration never merges (merge threshold 1.0)
    """
    reconciler = SpeakerReconciler()
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    _run(reconciler, 5.0, 10.0, "B", _voice(80))
    _run(reconciler, 10.0, 15.0, "B", _voice(5))
    assert not reconciler.last_merges
    assert reconciler.labels_minted == 2


def test_minting_time_survives_an_exported_state():
    """
    Test the minting time travels with the exported state, so the age
    limit still applies after a reconnect rebuilds the reconciler
    """
    reconciler = SpeakerReconciler(config=_config())
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    _run(reconciler, 5.0, 10.0, "B", _voice(70))
    state = reconciler.export_state()
    assert [s.minted_at for s in state.speakers] == [5.0, 10.0]

    continued = SpeakerReconciler(config=_config(), state=state)
    assert [s.minted_at for s in continued.speakers] == [5.0, 10.0]
