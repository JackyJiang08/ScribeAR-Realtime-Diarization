"""
Synthetic classroom hard case: one dominant speaker (the instructor) for
about 80 percent of the time, interrupted by short questions (5 to 15 s)
from three other voices, one of them buried in pink noise at 5 dB SNR.

Built from the AMI corpus (single distant microphone Array1-01, CC BY 4.0;
pyannote/AMI-diarization-setup only_words references) so every voice is a
real far-field recording from the same room and microphone: the four
people of the ES2004 series. Material is taken from stretches where only
one person speaks (no overlap in the reference), joined with short
silences; the speaker who has the most such speech over the series is the
instructor, the other three ask the questions. The layout, the question
lengths and the noisy questioner are fixed by the seed in the manifest, so
the case is the same on every machine. Nothing under data/ is committed.

Output (benchmarks/diarization/data/classroom/, gitignored):
  classroom.wav / .rttm / .uem   the case, scored like any other file
  classroom.questions.json       every question with its span, speaker and
                                 whether it carries noise, for
                                 classroom_score.py

Usage (from transcription_service/):
    uv run python benchmarks/diarization/prepare_classroom_case.py
"""

# pylint: disable=missing-function-docstring,too-many-locals

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ami_download import fetch_meeting  # noqa: E402
from bench_common import (  # noqa: E402
    BENCH_DIR,
    DATA_DIR,
    SAMPLE_RATE,
    load_audio,
    rttm_turns,
    write_rttm,
    write_uem,
)
from prepare_hard_cases import pink_noise  # noqa: E402

MANIFEST = BENCH_DIR / "classroom_case.json"
OUT_DIR = DATA_DIR / "classroom"


def exclusive_runs(turns, min_run_sec: float, join_gap_sec: float):
    """
    Per speaker, stretches where nobody else speaks: turns that no other
    speaker's turn overlaps, with same-speaker turns closer than
    `join_gap_sec` joined, at least `min_run_sec` long
    """
    runs: dict[str, list[tuple[float, float]]] = {}
    for start, end, speaker in turns:
        if any(
            os < end and oe > start and other != speaker
            for os, oe, other in turns
        ):
            continue
        mine = runs.setdefault(speaker, [])
        if mine and start - mine[-1][1] <= join_gap_sec:
            mine[-1] = (mine[-1][0], max(mine[-1][1], end))
        else:
            mine.append((start, end))
    return {
        speaker: [r for r in rs if r[1] - r[0] >= min_run_sec]
        for speaker, rs in runs.items()
    }


class Material:
    """
    Audio of one speaker's exclusive runs across meetings, consumed in
    order
    """

    def __init__(self):
        self.pieces: list[tuple[np.ndarray, str]] = []
        self.index = 0
        self.offset = 0

    def add(self, samples: np.ndarray, source: str):
        self.pieces.append((samples, source))

    @property
    def total_sec(self) -> float:
        return sum(len(p) for p, _ in self.pieces) / SAMPLE_RATE

    def take(self, seconds: float) -> np.ndarray:
        """
        The next `seconds` of material, whole runs first, the last run cut
        at the target; raises when the material is exhausted
        """
        wanted = int(seconds * SAMPLE_RATE)
        out = []
        got = 0
        while got < wanted:
            if self.index >= len(self.pieces):
                raise SystemExit("not enough exclusive speech for the case")
            piece = self.pieces[self.index][0][self.offset :]
            need = wanted - got
            if len(piece) <= need:
                out.append(piece)
                got += len(piece)
                self.index += 1
                self.offset = 0
            else:
                out.append(piece[:need])
                got += need
                self.offset += need
        return np.concatenate(out)


def fade(samples: np.ndarray, ms: float = 20.0) -> np.ndarray:
    n = min(len(samples) // 2, int(SAMPLE_RATE * ms / 1000))
    if n <= 0:
        return samples
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = samples.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def add_pink_noise(speech: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    power = float(np.mean(speech.astype(np.float64) ** 2)) or 1e-8
    noise = pink_noise(len(speech), seed) * np.sqrt(
        power / (10 ** (snr_db / 10))
    )
    mixed = speech + noise.astype(np.float32)
    peak = float(np.max(np.abs(mixed))) or 1.0
    return (mixed * (0.99 / peak) if peak > 0.99 else mixed).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--ami-dir", default=str(DATA_DIR / "ami"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_out = out_dir / "classroom.wav"
    if wav_out.exists() and not args.force:
        print("classroom: exists, skipping (use --force to redo)")
        return

    layout = manifest["layout"]
    rng = np.random.default_rng(int(manifest["seed"]))
    ami_dir = Path(args.ami_dir)

    # Material per speaker from every meeting of the series
    material: dict[str, Material] = {}
    for meeting in manifest["meetings"]:
        wav, rttm, _ = fetch_meeting(ami_dir, meeting)
        samples = load_audio(wav)
        runs = exclusive_runs(
            rttm_turns(rttm), layout["min_run_sec"], layout["join_gap_sec"]
        )
        for speaker, stretches in runs.items():
            for start, end in stretches:
                material.setdefault(speaker, Material()).add(
                    fade(
                        samples[
                            int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)
                        ]
                    ),
                    f"{meeting} {start:.1f}-{end:.1f}",
                )
    ranked = sorted(material, key=lambda s: -material[s].total_sec)
    instructor = ranked[0]
    questioners = ranked[1:4]
    print(
        "exclusive speech per speaker: "
        + ", ".join(f"{s} {material[s].total_sec:.0f}s" for s in ranked)
    )
    print(f"instructor {instructor}; questioners {questioners}")

    noisy = questioners[int(layout["noisy_questioner_index"])]
    gap = np.zeros(int(layout["gap_sec"] * SAMPLE_RATE), dtype=np.float32)
    lecture_sec = float(layout["lecture_block_sec"])
    per_questioner = int(layout["questions_per_questioner"])
    lengths = rng.uniform(
        layout["question_min_sec"],
        layout["question_max_sec"],
        size=len(questioners) * per_questioner,
    )
    order = [q for q in questioners for _ in range(per_questioner)]
    rng.shuffle(order)

    pieces: list[np.ndarray] = []
    turns: list[tuple[float, float, str]] = []
    questions = []
    cursor = 0.0

    def append(samples: np.ndarray, speaker: str) -> tuple[float, float]:
        nonlocal cursor
        pieces.append(samples)
        start = cursor
        cursor += len(samples) / SAMPLE_RATE
        turns.append((start, cursor, speaker))
        pieces.append(gap)
        cursor += len(gap) / SAMPLE_RATE
        return start, cursor - len(gap) / SAMPLE_RATE

    seed_noise = int(manifest["seed"])
    for index, (speaker, length) in enumerate(zip(order, lengths)):
        append(material[instructor].take(lecture_sec), instructor)
        speech = material[speaker].take(float(length))
        is_noisy = speaker == noisy
        if is_noisy:
            speech = add_pink_noise(
                speech, float(layout["noise_snr_db"]), seed_noise + index
            )
        start, end = append(speech, speaker)
        questions.append(
            {
                "index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "noisy": is_noisy,
                "length_sec": round(end - start, 2),
            }
        )
    append(material[instructor].take(lecture_sec), instructor)

    audio = np.concatenate(pieces).astype(np.float32)
    sf.write(str(wav_out), audio, SAMPLE_RATE, subtype="PCM_16")
    write_rttm(out_dir / "classroom.rttm", turns, "classroom")
    write_uem(out_dir / "classroom.uem", 0.0, cursor, "classroom")
    total = cursor
    instructor_sec = sum(e - s for s, e, sp in turns if sp == instructor)
    question_sec = sum(q["length_sec"] for q in questions)
    (out_dir / "classroom.questions.json").write_text(
        json.dumps(
            {
                "instructor": instructor,
                "questioners": questioners,
                "noisy_questioner": noisy,
                "duration_sec": round(total, 1),
                "instructor_fraction_of_speech": round(
                    instructor_sec / (instructor_sec + question_sec), 3
                ),
                "questions": questions,
                "manifest": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"classroom: {total:.0f} s, instructor {instructor_sec:.0f} s "
        f"({100 * instructor_sec / (instructor_sec + question_sec):.0f}% of speech), "
        f"{len(questions)} questions totalling {question_sec:.0f} s, "
        f"noisy questioner {noisy} at {layout['noise_snr_db']} dB"
    )


if __name__ == "__main__":
    main()
