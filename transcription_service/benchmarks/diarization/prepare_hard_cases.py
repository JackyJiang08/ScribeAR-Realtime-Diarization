"""
Materialises the hard-case set described in hard_cases.json: downloads the
AMI meetings it references, crops each case window (audio, RTTM, UEM) and
synthesises the noise case. Output: benchmarks/diarization/data/hard_cases/
(gitignored; audio is never committed).

Run the cases with:
  uv run python benchmarks/diarization/benchmark_baseline.py \
      --data benchmarks/diarization/data/hard_cases --suffix "" \
      --stream-sec 0 --skip-offline --out results/hard_cases.json
or `make benchmark_diarization_hardcases`.
"""

# pylint: disable=missing-function-docstring

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ami_download import (  # noqa: E402
    crop_regions,
    crop_turns,
    crop_wav,
    fetch_meeting,
    read_uem,
)
from bench_common import (  # noqa: E402
    BENCH_DIR,
    DATA_DIR,
    SAMPLE_RATE,
    load_audio,
    rttm_turns,
    write_rttm,
    write_uem,
)


def pink_noise(length: int, seed: int) -> np.ndarray:
    """1/f-shaped Gaussian noise, unit RMS."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=length).astype(np.float32)
    return pink / (np.sqrt(np.mean(pink**2)) + 1e-12)


def mix_noise(clean: np.ndarray, turns, snr_db: float, seed: int) -> np.ndarray:
    """Adds pink noise so that speech power / noise power = snr_db."""
    mask = np.zeros(len(clean), dtype=bool)
    for start, end, _ in turns:
        mask[int(start * SAMPLE_RATE) : int(end * SAMPLE_RATE)] = True
    speech = clean[mask] if mask.any() else clean
    speech_power = float(np.mean(speech.astype(np.float64) ** 2)) or 1e-8
    noise_power = speech_power / (10 ** (snr_db / 10))
    noise = pink_noise(len(clean), seed) * np.sqrt(noise_power)
    mixed = clean + noise.astype(np.float32)
    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default=str(BENCH_DIR / "hard_cases.json")
    )
    parser.add_argument("--ami-dir", default=str(DATA_DIR / "ami"))
    parser.add_argument("--out-dir", default=str(DATA_DIR / "hard_cases"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ami_dir = Path(args.ami_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in manifest["cases"]:
        name = case["name"]
        wav_out = out_dir / f"{name}.wav"
        if wav_out.exists() and not args.force:
            print(f"{name}: exists, skipping (use --force to redo)")
            continue
        wav, rttm, uem = fetch_meeting(ami_dir, case["meeting"])
        start, end = float(case["start_sec"]), float(case["end_sec"])
        duration = end - start
        turns = crop_turns(rttm_turns(rttm), start, end)
        regions = crop_regions(read_uem(uem), start, end)

        crop_wav(wav, wav_out, start, duration)
        noise = case.get("noise")
        if noise:
            clean = load_audio(wav_out)
            mixed = mix_noise(
                clean, turns, float(noise["snr_db"]), int(noise["seed"])
            )
            sf.write(str(wav_out), mixed, SAMPLE_RATE, subtype="PCM_16")
        write_rttm(out_dir / f"{name}.rttm", turns, name)
        # UEM: a single region per crop keeps scoring simple; AMI UEMs cover
        # the whole meeting anyway
        write_uem(out_dir / f"{name}.uem", regions[0][0], regions[-1][1], name)
        speakers = sorted({t[2] for t in turns})
        print(
            f"{name}: {case['meeting']} {start:.1f}-{end:.1f}s, "
            f"{len(turns)} turns, {len(speakers)} speakers"
            + (
                f", noise {noise['type']} SNR {noise['snr_db']} dB"
                if noise
                else ""
            )
        )
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nPrepared {len(manifest['cases'])} cases in {out_dir}")


if __name__ == "__main__":
    main()
