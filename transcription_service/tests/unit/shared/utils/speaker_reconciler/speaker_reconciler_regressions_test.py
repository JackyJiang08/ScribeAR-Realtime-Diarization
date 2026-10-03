"""
Regression tests for the SpeakerReconciler failure modes found by the
production audit (docs/diarization_production_audit.md, section 3.2 and
5.3). Each case replays a synthetic sequence of diarization runs through the
real class and asserts the behaviour a viewer needs: one voice keeps one
label.

These were expected-fail until Phase 2b; the embedding memory makes them
pass. Each run carries synthetic speaker embeddings the way the pyannote
context reports them (one unit vector per raw label): the same voice is the
same vector, different voices are orthogonal, so "clearly far" is exact.
"""

import numpy as np

from src.shared.utils.speaker_reconciler import (
    SpeakerReconciler,
    SpeakerReconcilerConfig,
    SpeakerSegment,
)

# The synthetic runs below are short, so the minting minimum is 1.5 s
# instead of the production default; everything else is the default.
SHORT_RUNS = SpeakerReconcilerConfig(min_mint_duration_sec=1.5)


def _reconciler() -> SpeakerReconciler:
    return SpeakerReconciler(config=SHORT_RUNS)


DIM = 8


def _voice(index: int) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


VOICE_A = _voice(0)
VOICE_B = _voice(1)
VOICE_C = _voice(2)
VOICE_D = _voice(3)


def _labels(segments: list[SpeakerSegment]) -> list[str]:
    return [segment.speaker for segment in segments]


def test_window_sliding_past_a_voice_keeps_its_label():
    """
    Buffer purge or force-finalize moved the window entirely past the
    previous run: the same single voice must keep spk_0, not become spk_1
    """
    reconciler = _reconciler()

    reconciler.reconcile(
        [SpeakerSegment(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}
    )
    result = reconciler.reconcile(
        [SpeakerSegment(12.0, 22.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}
    )

    assert _labels(result) == ["spk_0"]
    assert reconciler.labels_minted == 1


def test_speaker_returning_after_a_long_gap_keeps_its_label():
    """
    A talks, B talks, B alone for longer than the window, A returns: A must
    come back as spk_0 and no third label may be minted
    """
    reconciler = _reconciler()

    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 10.0, "SPEAKER_00"),  # A
            SpeakerSegment(10.0, 20.0, "SPEAKER_01"),  # B
        ],
        {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
    )
    reconciler.reconcile(
        [SpeakerSegment(30.0, 60.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_B}
    )  # B alone
    result = reconciler.reconcile(
        [
            SpeakerSegment(40.0, 62.0, "SPEAKER_01"),  # B continues
            SpeakerSegment(62.0, 70.0, "SPEAKER_00"),  # A returns
        ],
        {"SPEAKER_01": VOICE_B, "SPEAKER_00": VOICE_A},
    )

    assert _labels(result) == ["spk_1", "spk_0"]
    assert reconciler.labels_minted == 2


def test_short_backchannels_do_not_mint_session_labels():
    """
    A 0.4 s "yeah" inside another speaker's turn must not become a new
    session speaker; short raw labels inherit or stay unattributed, and only
    speech with enough cumulative duration mints a label
    """
    reconciler = _reconciler()

    reconciler.reconcile(
        [SpeakerSegment(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}
    )
    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 15.0, "SPEAKER_00"),
            SpeakerSegment(7.0, 7.4, "SPEAKER_01"),  # backchannel
        ],
        {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_C},
    )
    result = reconciler.reconcile(
        [
            SpeakerSegment(5.0, 20.0, "SPEAKER_00"),
            SpeakerSegment(12.0, 12.4, "SPEAKER_02"),  # another backchannel
        ],
        {"SPEAKER_00": VOICE_A, "SPEAKER_02": VOICE_D},
    )

    assert reconciler.labels_minted == 1
    assert result[0].speaker == "spk_0"
    # The backchannel sits inside A's turn, so by time overlap it inherits
    # A's label rather than minting; it never becomes a speaker of its own
    assert set(_labels(result)) == {"spk_0"}


def test_a_short_voice_mints_once_it_has_spoken_enough():
    """
    A voice that only ever speaks in short bursts still gets a label once
    its bursts add up to the minimum, and keeps it afterwards
    """
    reconciler = _reconciler()
    reconciler.reconcile(
        [SpeakerSegment(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}
    )
    for start in (11.0, 16.0, 21.0):
        result = reconciler.reconcile(
            [
                SpeakerSegment(start - 1.0, start + 4.0, "SPEAKER_00"),
                SpeakerSegment(start, start + 0.6, "SPEAKER_01"),
            ],
            {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
        )

    assert reconciler.labels_minted == 2
    assert _labels(result) == ["spk_0", "spk_1"]


def test_segmentation_split_then_merge_does_not_ping_pong():
    """
    The diarizer alternates between seeing one voice and splitting it in
    two (A; A+B; A; A+B): labels must settle on at most two session labels
    and A's long turn must stay spk_0 throughout
    """
    reconciler = _reconciler()

    runs = [
        ([SpeakerSegment(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}),
        (
            [
                SpeakerSegment(0.0, 12.0, "SPEAKER_00"),
                SpeakerSegment(12.0, 15.0, "SPEAKER_01"),
            ],
            {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
        ),
        ([SpeakerSegment(5.0, 20.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}),
        (
            [
                SpeakerSegment(10.0, 25.0, "SPEAKER_00"),
                SpeakerSegment(22.0, 25.0, "SPEAKER_01"),
            ],
            {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
        ),
    ]
    for segments, embeddings in runs:
        result = reconciler.reconcile(segments, embeddings)
        assert result[0].speaker == "spk_0"

    assert reconciler.labels_minted <= 2


def test_merged_speakers_split_again_reuse_their_labels():
    """
    Two established speakers are merged into one raw label for a run, then
    split again: the split must return spk_0 and spk_1, not mint spk_2
    """
    reconciler = _reconciler()

    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 5.0, "SPEAKER_00"),  # A
            SpeakerSegment(5.0, 10.0, "SPEAKER_01"),  # B
        ],
        {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
    )
    # The merged cluster's centroid is a blend of both voices
    reconciler.reconcile(
        [SpeakerSegment(0.0, 15.0, "SPEAKER_00")],
        {"SPEAKER_00": VOICE_A * 0.8 + VOICE_B * 0.6},
    )
    result = reconciler.reconcile(
        [
            SpeakerSegment(5.0, 18.0, "SPEAKER_00"),  # A
            SpeakerSegment(18.0, 20.0, "SPEAKER_01"),  # B back
        ],
        {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
    )

    assert sorted(_labels(result)) == ["spk_0", "spk_1"]
    assert reconciler.labels_minted == 2


def test_reconnect_continues_labels_instead_of_restarting_at_spk_0():
    """
    node-server reconnects to the Python service after any upstream blip and
    the provider builds a new job and a new reconciler. The rebuilt
    reconciler, given the exported state, must continue the session's
    labels (A stays spk_0, B spk_1) rather than relabelling the same people
    from spk_0.
    """
    first = _reconciler()
    first.reconcile(
        [
            SpeakerSegment(0.0, 10.0, "SPEAKER_00"),  # A
            SpeakerSegment(10.0, 20.0, "SPEAKER_01"),  # B
        ],
        {"SPEAKER_00": VOICE_A, "SPEAKER_01": VOICE_B},
    )

    rebuilt = SpeakerReconciler(state=first.export_state())
    result = rebuilt.reconcile(
        [
            SpeakerSegment(20.0, 25.0, "SPEAKER_00"),  # B keeps talking
            SpeakerSegment(25.0, 30.0, "SPEAKER_01"),  # A again
        ],
        {"SPEAKER_00": VOICE_B, "SPEAKER_01": VOICE_A},
    )

    assert _labels(result) == ["spk_1", "spk_0"]
    assert rebuilt.labels_minted == 2


def test_exported_state_is_a_copy():
    """
    Mutating the rebuilt reconciler never touches the original's memory
    """
    first = _reconciler()
    first.reconcile(
        [SpeakerSegment(0.0, 10.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_A}
    )
    rebuilt = SpeakerReconciler(state=first.export_state())
    rebuilt.reconcile(
        [SpeakerSegment(10.0, 20.0, "SPEAKER_00")], {"SPEAKER_00": VOICE_B}
    )

    assert first.labels_minted == 1
    assert rebuilt.labels_minted == 2
