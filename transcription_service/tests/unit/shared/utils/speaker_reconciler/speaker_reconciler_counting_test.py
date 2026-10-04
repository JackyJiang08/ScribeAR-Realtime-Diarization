"""
Unit tests for the under-counting guards of SpeakerReconciler: the
sustained-voice split and the periodic session-level re-clustering
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


def _run(reconciler, start, end, raw, voice, scale=1.0):
    return reconciler.reconcile(
        [SpeakerSegment(start, end, raw)], {raw: voice * scale}
    )


def test_a_near_voice_attaches_until_it_has_sustained_long_enough():
    """
    Test a voice scoring between the new-speaker and the match threshold
    attaches to the speaker it resembles at first and splits off once it
    has `sustained_split_sec` of evidence, without touching earlier labels
    """
    config = SpeakerReconcilerConfig(
        match_threshold=0.9,
        new_speaker_threshold=0.5,
        attach_threshold=0.2,
        min_mint_duration_sec=1.0,
        sustained_split_sec=8.0,
        overlap_bonus=0.0,
    )
    reconciler = SpeakerReconciler(config=config)
    _run(reconciler, 0.0, 5.0, "A", _voice(0))

    # cos(40 deg) = 0.77: near spk_0 but not a clear match; 4 s per pass
    first = _run(reconciler, 5.0, 9.0, "B", _voice(40))
    assert [s.speaker for s in first] == ["spk_0"]
    assert reconciler.labels_minted == 1

    second = _run(reconciler, 9.0, 13.0, "B", _voice(40))
    assert [s.speaker for s in second] == ["spk_1"]
    assert reconciler.labels_minted == 2
    # The earlier pass was not relabelled: splits rule from now on only
    assert not reconciler.last_merges


def test_sustained_split_is_off_by_default():
    """
    Test the default configuration keeps attaching a near voice
    """
    config = SpeakerReconcilerConfig(
        match_threshold=0.9,
        new_speaker_threshold=0.5,
        attach_threshold=0.2,
        min_mint_duration_sec=1.0,
        overlap_bonus=0.0,
    )
    reconciler = SpeakerReconciler(config=config)
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    for start in (5.0, 9.0, 13.0, 17.0):
        result = _run(reconciler, start, start + 4.0, "B", _voice(40))
    assert [s.speaker for s in result] == ["spk_0"]
    assert reconciler.labels_minted == 1


def test_a_clear_match_never_splits():
    """
    Test a voice at or above the match threshold is never pooled as a
    sustained split candidate however long it speaks
    """
    config = SpeakerReconcilerConfig(
        match_threshold=0.6,
        new_speaker_threshold=0.3,
        min_mint_duration_sec=1.0,
        sustained_split_sec=4.0,
        overlap_bonus=0.0,
    )
    reconciler = SpeakerReconciler(config=config)
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    for start in (5.0, 10.0, 15.0, 20.0):
        result = _run(reconciler, start, start + 5.0, "A", _voice(10))
    assert [s.speaker for s in result] == ["spk_0"]
    assert reconciler.labels_minted == 1


def _recluster_config(**overrides):
    base = {
        "match_threshold": 0.6,
        "new_speaker_threshold": 0.3,
        "attach_threshold": 0.2,
        "min_mint_duration_sec": 1.0,
        "min_update_sec": 0.0,
        "overlap_bonus": 0.0,
        "recluster_period_sec": 10.0,
        "recluster_min_split_sec": 4.0,
    }
    base.update(overrides)
    return SpeakerReconcilerConfig(**base)


def test_reclustering_merges_speakers_the_clustering_joins():
    """
    Test two session labels whose tracks the clusterer puts in one cluster
    are merged (junior onto senior) and reported through last_merges
    """

    def one_cluster(embeddings, seconds):
        del seconds
        return np.zeros(len(embeddings), dtype=int)

    reconciler = SpeakerReconciler(
        config=_recluster_config(), clusterer=one_cluster
    )
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    # cos(100 deg) < 0.3: clearly a new speaker by cosine
    _run(reconciler, 5.0, 9.0, "B", _voice(100))
    assert reconciler.labels_minted == 2

    # The pass that crosses the 10 s period triggers the re-clustering
    result = _run(reconciler, 9.0, 12.0, "B", _voice(100))
    assert reconciler.reclusterings == 1
    assert reconciler.last_merges == [("spk_1", "spk_0")]
    assert [s.speaker for s in result] == ["spk_0"]
    assert [s.label for s in reconciler.speakers] == ["spk_0"]


def test_reclustering_splits_a_speaker_whose_tracks_fall_apart():
    """
    Test a speaker whose tracks the clusterer separates into two clusters
    (each long enough) keeps its label for the voice with most of its
    speech and gives the other voice a child label, from the next pass on
    """
    calls = []

    def by_sign(embeddings, seconds):
        del seconds
        calls.append(len(embeddings))
        return (np.asarray(embeddings)[:, 1] > 0.5).astype(int)

    reconciler = SpeakerReconciler(
        config=_recluster_config(), clusterer=by_sign
    )
    for start in (0.0, 5.0, 10.0):
        _run(reconciler, start, start + 5.0, "A", _voice(0))
    # A second voice 40 degrees away (cos 0.77 >= match): the same label,
    # by design of the cosine rule
    _run(reconciler, 15.0, 16.5, "B", _voice(40))
    assert reconciler.labels_minted == 1
    # The re-clustering at 10 s saw one voice; the one at 21.5 s sees 6.5 s
    # of the second voice, enough to split it off
    assert reconciler.reclusterings == 1
    result = _run(reconciler, 16.5, 21.5, "B", _voice(40))
    assert reconciler.reclusterings == 2
    assert reconciler.last_splits == [("spk_0", "spk_1")]
    assert [s.label for s in reconciler.speakers] == ["spk_0", "spk_1"]
    # The split rules from the next pass: this pass's labels stand
    assert [s.speaker for s in result] == ["spk_0"]
    following = _run(reconciler, 21.5, 26.5, "B", _voice(40))
    assert [s.speaker for s in following] == ["spk_1"]
    # The first voice, with most of the speech, kept the original label
    again = _run(reconciler, 26.5, 31.5, "A", _voice(0))
    assert [s.speaker for s in again] == ["spk_0"]
    assert calls[:2] == [2, 5], "the clusterer saw every track each time"


def test_reclustering_is_skipped_without_a_clusterer_or_when_disabled():
    """
    Test a reconciler without a clusterer, or with the period at 0, never
    re-clusters and keeps no track history
    """
    reconciler = SpeakerReconciler(config=_recluster_config())
    for start in range(0, 30, 5):
        _run(reconciler, start, start + 5.0, "A", _voice(0))
    assert reconciler.reclusterings == 0
    assert reconciler.export_state().tracks  # history kept, clusterer absent

    disabled = SpeakerReconciler(
        config=_recluster_config(recluster_period_sec=0.0),
        clusterer=lambda e, s: np.zeros(len(e), dtype=int),
    )
    for start in range(0, 30, 5):
        _run(disabled, start, start + 5.0, "A", _voice(0))
    assert disabled.reclusterings == 0
    assert not disabled.export_state().tracks


def test_exported_state_carries_tracks_and_shadows_as_copies():
    """
    Test the track history and the near-voice pool survive a state export
    and the copies are independent of the original
    """
    config = _recluster_config(sustained_split_sec=100.0)
    reconciler = SpeakerReconciler(config=config)
    _run(reconciler, 0.0, 5.0, "A", _voice(0))
    _run(reconciler, 5.0, 9.0, "B", _voice(65))  # near voice -> shadow
    state = reconciler.export_state()
    assert len(state.tracks) == 2
    assert len(state.shadows) == 1
    state.tracks[0].embedding[:] = 0.0
    assert reconciler.export_state().tracks[0].embedding[0] != 0.0
    continued = SpeakerReconciler(config=config, state=state)
    assert continued.labels_minted == 1
