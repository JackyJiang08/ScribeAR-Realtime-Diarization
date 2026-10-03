"""
Selects the hard-case windows from AMI reference RTTMs and writes the
manifest `hard_cases.json`.

Run once when the case set is (re)defined; the manifest is committed so the
cases are fixed and reproducible, and prepare_hard_cases.py materialises the
audio from it. The selection is purely from the reference annotations, so
it needs no audio and no model:

  overlap       - the 120 s window with the largest fraction of time where
                  two or more reference speakers talk at once
  short_turns   - the 120 s window with the most reference turns under 1 s
                  (backchannels), low overlap preferred
  return_after_gap - a speaker who talks, is silent for longer than the
                  30 s rolling buffer, and returns: the window from 30 s
                  before their last turn to 30 s after their return
  four_speakers - the 120 s window in which the most reference speakers
                  each have at least 10 s of speech
  noise_*       - a clean 120 s window plus pink noise mixed at a fixed
                  SNR by prepare_hard_cases.py (clean counterpart kept)

Usage (from transcription_service/, after prepare_ami_baseline.sh):
  uv run python benchmarks/diarization/select_hard_cases.py
"""

# pylint: disable=missing-function-docstring,too-many-locals

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import BENCH_DIR, DATA_DIR, rttm_turns  # noqa: E402

WINDOW = 120.0
STEP = 30.0
MIN_GAP_SEC = 60.0  # twice the 30 s buffer: the window has certainly slid past
MAX_CASE_SEC = 360.0  # keep the return case replayable in a few minutes


def overlap_seconds(turns, start, end) -> float:
    events = []
    for s, e, _ in turns:
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            events.append((s2, 1))
            events.append((e2, -1))
    events.sort()
    active = 0
    last = start
    overlap = 0.0
    for t, delta in events:
        if active >= 2:
            overlap += t - last
        last = t
        active += delta
    return overlap


def speech_per_speaker(turns, start, end) -> dict[str, float]:
    totals: dict[str, float] = {}
    for s, e, spk in turns:
        s2, e2 = max(s, start), min(e, end)
        if e2 > s2:
            totals[spk] = totals.get(spk, 0.0) + (e2 - s2)
    return totals


def windows(duration):
    start = 0.0
    while start + WINDOW <= duration:
        yield start, start + WINDOW
        start += STEP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_DIR / "ami"))
    parser.add_argument(
        "--meetings", nargs="*", default=["ES2004a", "IS1009a", "TS3003a"]
    )
    parser.add_argument("--out", default=str(BENCH_DIR / "hard_cases.json"))
    args = parser.parse_args()

    data = Path(args.data)
    stats = {}
    for meeting in args.meetings:
        rttm = data / f"{meeting}.rttm"
        if not rttm.exists():
            raise SystemExit(
                f"{rttm} missing; run prepare_ami_baseline.sh first"
            )
        turns = rttm_turns(rttm)
        duration = max(e for _, e, _ in turns)
        stats[meeting] = (turns, duration)

    best_overlap = (-1.0, None)
    best_short = (-1, None)
    best_gap = (-1.0, None)
    best_four = (-1, None)
    for meeting, (turns, duration) in stats.items():
        for start, end in windows(duration):
            ov = overlap_seconds(turns, start, end) / WINDOW
            short = sum(
                1 for s, e, _ in turns if start <= s < end and e - s < 1.0
            )
            per_spk = speech_per_speaker(turns, start, end)
            busy = sum(1 for v in per_spk.values() if v >= 10.0)
            balance = min(per_spk.values()) if per_spk else 0.0
            if ov > best_overlap[0]:
                best_overlap = (
                    ov,
                    (meeting, start, end, {"overlap_fraction": round(ov, 3)}),
                )
            # Short turns: prefer many short turns in a window with little
            # overlap so the case is distinct from the overlap case.
            score_short = short - 40 * ov
            if score_short > best_short[0]:
                best_short = (
                    score_short,
                    (
                        meeting,
                        start,
                        end,
                        {
                            "short_turns": short,
                            "overlap_fraction": round(ov, 3),
                        },
                    ),
                )
            score_four = busy * 1000 + balance
            if score_four > best_four[0]:
                best_four = (
                    score_four,
                    (
                        meeting,
                        start,
                        end,
                        {
                            "speakers_with_10s": busy,
                            "least_speech_sec": round(balance, 1),
                        },
                    ),
                )
        for speaker in sorted({t[2] for t in turns}):
            own = [(s, e) for s, e, spk in turns if spk == speaker]
            for (s1, e1), (s2, _) in zip(own, own[1:]):
                gap = s2 - e1
                start = max(0.0, e1 - 30.0)
                end = min(duration, s2 + 30.0)
                if (
                    gap >= MIN_GAP_SEC
                    and end - start <= MAX_CASE_SEC
                    and gap > best_gap[0]
                ):
                    best_gap = (
                        gap,
                        (
                            meeting,
                            start,
                            end,
                            {
                                "speaker": speaker,
                                "gap_sec": round(gap, 1),
                                "leaves_at": round(e1, 1),
                                "returns_at": round(s2, 1),
                            },
                        ),
                    )

    def case(name, picked, description, extra=None):
        meeting, start, end, info = picked
        return {
            "name": name,
            "meeting": meeting,
            "start_sec": round(start, 1),
            "end_sec": round(end, 1),
            "description": description,
            "selection": info,
            **(extra or {}),
        }

    cases = [
        case(
            "overlap",
            best_overlap[1],
            "largest fraction of overlapping speech in a 120 s window",
        ),
        case(
            "short_turns",
            best_short[1],
            "most reference turns under 1 s (backchannels) in a 120 s window",
        ),
        case(
            "return_after_gap",
            best_gap[1],
            "a speaker silent for longer than the rolling buffer returns; window spans 30 s before leaving to 30 s after returning",
        ),
        case(
            "four_speakers",
            best_four[1],
            "120 s window in which the most reference speakers each have at least 10 s of speech",
        ),
        case(
            "noise_clean",
            best_short[1],
            "clean counterpart of the noise case (same audio, no added noise)",
        ),
        case(
            "noise_pink_snr5",
            best_short[1],
            "same window with pink noise mixed at 5 dB SNR (speech power measured over reference speech regions), seed 20261002",
            {"noise": {"type": "pink", "snr_db": 5.0, "seed": 20261002}},
        ),
    ]
    manifest = {
        "_comment": [
            "Hard-case set for the diarization benchmark. Audio is never",
            "committed: prepare_hard_cases.py downloads the AMI meetings",
            "(single distant microphone Array1-01, AMI corpus mirror,",
            "CC BY 4.0) and the pyannote/AMI-diarization-setup only_words",
            "references, crops each window and synthesises the noise case.",
            "Generated by select_hard_cases.py from the reference RTTMs;",
            "regenerate only when deliberately changing the case set.",
        ],
        "sources": {
            "audio": "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/<meeting>/audio/<meeting>.Array1-01.wav",
            "references": "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/main/only_words/rttms/test/<meeting>.rttm (+ uems/test)",
            "noise": "synthesised pink noise (1/f spectrum shaped from seeded Gaussian noise), prepare_hard_cases.py",
        },
        "cases": cases,
    }
    Path(args.out).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(cases, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
