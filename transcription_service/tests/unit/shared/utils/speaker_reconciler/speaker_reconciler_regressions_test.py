"""
Regression tests for the SpeakerReconciler failure modes found by the
production audit (docs/diarization_production_audit.md, section 3.2 and
5.3). Each case replays a synthetic sequence of diarization runs through the
real class and asserts the behaviour a viewer needs: one voice keeps one
label.

Every test is marked expected-fail (strict) because the current
overlap-voting reconciler mints a fresh label whenever a raw label has no
overlapping predecessor. Once the Phase 2b fix lands (speaker-embedding
memory with a minimum duration before minting, and label continuity across
reconnects) these tests must flip to passing, at which point the markers
come off. A test that starts passing early is reported as a failure by
`strict=True`, so a partial fix cannot go unnoticed either.

The eventual fix will most likely carry speaker embeddings on the segments
the reconciler receives; these tests then need synthetic embeddings added to
their inputs (same voice = same vector) and the presumed state API in the
reconnect test replaced by the real one.
"""

import pytest

from src.shared.utils.speaker_reconciler import (
    SpeakerReconciler,
    SpeakerSegment,
)

PHASE_2B = (
    "audit 3.2: overlap voting cannot recognise a voice without a shared "
    "window; needs embedding memory (Phase 2b)"
)
RECONNECT = (
    "audit 5.3: every reconnect builds a fresh reconciler, so labels restart "
    "at spk_0; needs state carried across the job rebuild (Phase 2b)"
)


def _labels(segments: list[SpeakerSegment]) -> list[str]:
    return [segment.speaker for segment in segments]


@pytest.mark.xfail(strict=True, reason=PHASE_2B)
def test_window_sliding_past_a_voice_keeps_its_label():
    """
    Buffer purge or force-finalize moved the window entirely past the
    previous run: the same single voice must keep spk_0, not become spk_1
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile([SpeakerSegment(0.0, 10.0, "SPEAKER_00")])
    result = reconciler.reconcile([SpeakerSegment(12.0, 22.0, "SPEAKER_00")])

    assert _labels(result) == ["spk_0"]
    assert reconciler.labels_minted == 1


@pytest.mark.xfail(strict=True, reason=PHASE_2B)
def test_speaker_returning_after_a_long_gap_keeps_its_label():
    """
    A talks, B talks, B alone for longer than the window, A returns: A must
    come back as spk_0 and no third label may be minted
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 10.0, "SPEAKER_00"),  # A
            SpeakerSegment(10.0, 20.0, "SPEAKER_01"),  # B
        ]
    )
    reconciler.reconcile([SpeakerSegment(30.0, 60.0, "SPEAKER_00")])  # B alone
    result = reconciler.reconcile(
        [
            SpeakerSegment(40.0, 62.0, "SPEAKER_01"),  # B continues
            SpeakerSegment(62.0, 70.0, "SPEAKER_00"),  # A returns
        ]
    )

    assert _labels(result) == ["spk_1", "spk_0"]
    assert reconciler.labels_minted == 2


@pytest.mark.xfail(strict=True, reason=PHASE_2B)
def test_short_backchannels_do_not_mint_session_labels():
    """
    A 0.4 s "yeah" inside another speaker's turn must not become a new
    session speaker; short raw labels inherit or stay unattributed, and only
    speech with enough cumulative duration mints a label
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile([SpeakerSegment(0.0, 10.0, "SPEAKER_00")])
    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 15.0, "SPEAKER_00"),
            SpeakerSegment(7.0, 7.4, "SPEAKER_01"),  # backchannel
        ]
    )
    result = reconciler.reconcile(
        [
            SpeakerSegment(5.0, 20.0, "SPEAKER_00"),
            SpeakerSegment(12.0, 12.4, "SPEAKER_02"),  # another backchannel
        ]
    )

    assert reconciler.labels_minted == 1
    assert result[0].speaker == "spk_0"


@pytest.mark.xfail(strict=True, reason=PHASE_2B)
def test_segmentation_split_then_merge_does_not_ping_pong():
    """
    The diarizer alternates between seeing one voice and splitting it in
    two (A; A+B; A; A+B): labels must settle on at most two session labels
    and A's long turn must stay spk_0 throughout
    """
    reconciler = SpeakerReconciler()

    runs = [
        [SpeakerSegment(0.0, 10.0, "SPEAKER_00")],
        [
            SpeakerSegment(0.0, 12.0, "SPEAKER_00"),
            SpeakerSegment(12.0, 15.0, "SPEAKER_01"),
        ],
        [SpeakerSegment(5.0, 20.0, "SPEAKER_00")],
        [
            SpeakerSegment(10.0, 25.0, "SPEAKER_00"),
            SpeakerSegment(22.0, 25.0, "SPEAKER_01"),
        ],
    ]
    for run in runs:
        result = reconciler.reconcile(run)
        assert result[0].speaker == "spk_0"

    assert reconciler.labels_minted <= 2


@pytest.mark.xfail(strict=True, reason=PHASE_2B)
def test_merged_speakers_split_again_reuse_their_labels():
    """
    Two established speakers are merged into one raw label for a run, then
    split again: the split must return spk_0 and spk_1, not mint spk_2
    """
    reconciler = SpeakerReconciler()

    reconciler.reconcile(
        [
            SpeakerSegment(0.0, 5.0, "SPEAKER_00"),  # A
            SpeakerSegment(5.0, 10.0, "SPEAKER_01"),  # B
        ]
    )
    reconciler.reconcile([SpeakerSegment(0.0, 15.0, "SPEAKER_00")])  # merged
    result = reconciler.reconcile(
        [
            SpeakerSegment(5.0, 18.0, "SPEAKER_00"),  # A
            SpeakerSegment(18.0, 20.0, "SPEAKER_01"),  # B back
        ]
    )

    assert sorted(_labels(result)) == ["spk_0", "spk_1"]
    assert reconciler.labels_minted == 2


@pytest.mark.xfail(strict=True, reason=RECONNECT)
def test_reconnect_continues_labels_instead_of_restarting_at_spk_0():
    """
    node-server reconnects to the Python service after any upstream blip and
    the provider builds a new job and a new reconciler. The rebuilt
    reconciler must continue the session's labels (A stays spk_0, B spk_1)
    rather than relabelling the same people from spk_0.

    Presumed API: `export_state()` / `SpeakerReconciler(state=...)`; replace
    with the real one when the fix lands.
    """
    first = SpeakerReconciler()
    first.reconcile(
        [
            SpeakerSegment(0.0, 10.0, "SPEAKER_00"),  # A
            SpeakerSegment(10.0, 20.0, "SPEAKER_01"),  # B
        ]
    )

    # pylint: disable=no-member,unexpected-keyword-arg
    rebuilt = SpeakerReconciler(state=first.export_state())
    result = rebuilt.reconcile(
        [
            SpeakerSegment(20.0, 25.0, "SPEAKER_00"),  # B keeps talking
            SpeakerSegment(25.0, 30.0, "SPEAKER_01"),  # A again
        ]
    )

    assert _labels(result) == ["spk_1", "spk_0"]
    assert rebuilt.labels_minted == 2
