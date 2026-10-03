"""
Builds the long-session soak stream: consecutive AMI meetings of the same
participants concatenated into one 16 kHz mono recording with a matching
RTTM, so a 60-minute (or 2-hour) session can be replayed against ground
truth. Output: benchmarks/diarization/data/soak/soak_<minutes>min.{wav,rttm,uem}
plus a MANIFEST json with every part's offset (gitignored; never committed).

Meetings, in order:
  ES2004a, ES2004b, ES2004c, ES2004d  (one four-person team, same voices
                                       across all four meetings, ~132 min)
  IS1009a, IS1009b, IS1009c, IS1009d  (another team, appended only when
                                       more than 132 min is requested)
AMI speaker ids are global, so a participant keeps one reference label
across meetings: that is what makes label drift and swaps measurable over
an hour.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/prepare_soak.py --minutes 60
  uv run python benchmarks/diarization/prepare_soak.py --minutes 120
"""

# pylint: disable=missing-function-docstring

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ami_download import crop_wav, fetch_meeting, read_uem  # noqa: E402
from bench_common import (  # noqa: E402
    DATA_DIR,
    SAMPLE_RATE,
    rttm_turns,
    write_rttm,
)

SERIES = [
    "ES2004a",
    "ES2004b",
    "ES2004c",
    "ES2004d",
    "IS1009a",
    "IS1009b",
    "IS1009c",
    "IS1009d",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=60, choices=[60, 120])
    parser.add_argument("--ami-dir", default=str(DATA_DIR / "ami"))
    parser.add_argument("--out-dir", default=str(DATA_DIR / "soak"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_sec = args.minutes * 60
    ami_dir = Path(args.ami_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"soak_{args.minutes}min"
    wav_out = out_dir / f"{stem}.wav"
    if wav_out.exists() and not args.force:
        print(f"{wav_out} exists; use --force to rebuild")
        return

    parts = []
    turns_all = []
    uem_regions = []
    offset = 0.0
    with sf.SoundFile(
        str(wav_out), "w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16"
    ) as out:
        for meeting in SERIES:
            if offset >= target_sec:
                break
            wav, rttm, uem = fetch_meeting(ami_dir, meeting)
            part = out_dir / "parts" / f"{meeting}.wav"
            if not part.exists():
                info = sf.info(str(wav))
                crop_wav(wav, part, 0.0, info.duration)
            samples, rate = sf.read(str(part), dtype="float32")
            assert rate == SAMPLE_RATE
            remaining = target_sec - offset
            take_sec = min(len(samples) / SAMPLE_RATE, remaining)
            take = samples[: int(take_sec * SAMPLE_RATE)]
            out.write(take)
            for start, end, speaker in rttm_turns(rttm):
                if start >= take_sec:
                    continue
                turns_all.append(
                    (start + offset, min(end, take_sec) + offset, speaker)
                )
            for start, end in read_uem(uem):
                if start < take_sec:
                    uem_regions.append(
                        (start + offset, min(end, take_sec) + offset)
                    )
            parts.append(
                {
                    "meeting": meeting,
                    "offset_sec": round(offset, 3),
                    "seconds": round(take_sec, 3),
                }
            )
            print(
                f"{meeting}: {take_sec / 60:.1f} min at offset {offset / 60:.1f} min"
            )
            offset += take_sec

    write_rttm(out_dir / f"{stem}.rttm", sorted(turns_all), stem)
    (out_dir / f"{stem}.uem").write_text(
        "".join(f"{stem} 1 {s:.3f} {e:.3f}\n" for s, e in uem_regions),
        encoding="utf-8",
    )
    speakers = sorted({t[2] for t in turns_all})
    manifest = {
        "stem": stem,
        "minutes": args.minutes,
        "total_sec": round(offset, 1),
        "parts": parts,
        "reference_speakers": speakers,
        "sources": {
            "audio": "AMI corpus mirror, Array1-01 single distant microphone (CC BY 4.0)",
            "references": "pyannote/AMI-diarization-setup only_words RTTM + UEM",
        },
    }
    (out_dir / f"{stem}.MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nWrote {wav_out} ({offset / 60:.1f} min, {len(turns_all)} turns, "
        f"{len(speakers)} speakers: {speakers})"
    )


if __name__ == "__main__":
    main()
