"""
Benchmark speaker diarization engines for ScribeAR's streaming pipeline.

Measures two things per engine:
  1. Full-file real-time factor (RTF): wall time / audio duration.
  2. Tick simulation: the pipeline re-processes a rolling buffer (max 30s)
     every 5s job tick, mirroring WhisperStreamingProviderJob. Each tick's
     diarization must comfortably fit inside the 5s budget (shared with
     Whisper) for live use.

Engines:
  pyannote   - pyannote/speaker-diarization-community-1 (needs HF token)
  sortformer - nvidia/diar_streaming_sortformer_4spk-v2 via NeMo (GPU advised)

Usage:
  python benchmark_diarization.py --audio sample.wav --engine pyannote --device cpu
  python benchmark_diarization.py --audio sample.wav --engine both --device cuda

Input audio must be 16 kHz mono WAV. Convert with:
  ffmpeg -i input.m4a -ac 1 -ar 16000 -acodec pcm_s16le sample.wav
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
TICK_SEC = 5.0
MAX_BUFFER_SEC = 30.0


def load_audio(path: str) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE:
        raise SystemExit(
            f"Expected {SAMPLE_RATE} Hz audio, got {rate} Hz. "
            "Convert with: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav"
        )
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def report(
    name: str, audio_sec: float, full_sec: float, tick_secs: list[float]
):
    print(f"\n=== {name} ===")
    print(f"Audio duration        : {audio_sec:8.1f} s")
    print(
        f"Full-file wall time   : {full_sec:8.1f} s"
        f"  (RTF {full_sec / audio_sec:.3f})"
    )
    if tick_secs:
        worst = max(tick_secs)
        mean = sum(tick_secs) / len(tick_secs)
        print(
            f"Tick sim ({len(tick_secs)} ticks)  :"
            f" mean {mean:6.2f} s, worst {worst:6.2f} s"
        )
        verdict = "FITS" if worst < TICK_SEC else "TOO SLOW for"
        print(f"Verdict               : {verdict} the {TICK_SEC:.0f}s job tick")


def tick_windows(samples: np.ndarray):
    """Yield rolling buffers as WhisperStreamingProviderJob would see them."""
    tick = int(TICK_SEC * SAMPLE_RATE)
    max_buf = int(MAX_BUFFER_SEC * SAMPLE_RATE)
    for end in range(tick, len(samples) + 1, tick):
        yield samples[max(0, end - max_buf) : end]


def bench_pyannote(samples: np.ndarray, device: str, hf_token: str | None):
    import torch
    from pyannote.audio import Pipeline

    token = hf_token or os.environ.get("HUGGINGFACE_ACCESS_TOKEN")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=token
    )
    pipeline.to(torch.device(device))

    def run(chunk: np.ndarray):
        waveform = torch.from_numpy(chunk).unsqueeze(0)
        return pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})

    run(samples[: SAMPLE_RATE * 5])  # warm up model weights and kernels

    start = time.perf_counter()
    output = run(samples)
    full_sec = time.perf_counter() - start

    diarization = getattr(
        output,
        "exclusive_speaker_diarization",
        getattr(output, "speaker_diarization", output),
    )
    speakers = {
        label for _, _, label in diarization.itertracks(yield_label=True)
    }
    print(f"pyannote found speakers: {sorted(speakers)}")

    ticks = []
    for window in tick_windows(samples):
        start = time.perf_counter()
        run(window)
        ticks.append(time.perf_counter() - start)

    report("pyannote community-1", len(samples) / SAMPLE_RATE, full_sec, ticks)


def bench_sortformer(samples: np.ndarray, device: str, audio_path: str):
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2"
    )
    model = model.to(device)
    model.eval()

    start = time.perf_counter()
    outputs = model.diarize(audio=[audio_path], batch_size=1)
    full_sec = time.perf_counter() - start

    print(f"sortformer segments (first 5): {outputs[0][:5]}")
    report("Streaming Sortformer v2", len(samples) / SAMPLE_RATE, full_sec, [])
    print(
        "Note: NeMo's offline diarize() is benchmarked here; true streaming "
        "uses chunked inference and will be measured during integration."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV file")
    parser.add_argument(
        "--engine", choices=["pyannote", "sortformer", "both"], default="both"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated pyannote models "
        "(defaults to HUGGINGFACE_ACCESS_TOKEN env var)",
    )
    args = parser.parse_args()

    samples = load_audio(args.audio)
    print(f"Loaded {args.audio}: {len(samples) / SAMPLE_RATE:.1f} s of audio")

    if args.engine in ("pyannote", "both"):
        bench_pyannote(samples, args.device, args.hf_token)
    if args.engine in ("sortformer", "both"):
        bench_sortformer(samples, args.device, args.audio)


if __name__ == "__main__":
    main()
