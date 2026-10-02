"""
Accuracy + speed baseline for ScribeAR's pyannote speaker diarization.

Runs against a folder of 16 kHz mono WAV files with matching reference
RTTM (and optional UEM) files, e.g. AMI meetings prepared by
prepare_ami_baseline.sh, and writes a JSON report.

Two measurements per file:
  1. offline   - one pipeline pass over the whole file. Gives the best-case
                 DER / JER, real-time factor and peak memory.
  2. streaming - replays the file the way WhisperStreamingProviderJob sees
                 it: every job tick (5 s) the rolling buffer (max 30 s) is
                 re-diarized and raw labels are passed through
                 SpeakerReconciler. Scores the labels a viewer would actually
                 see, plus label latency, label flips and per-tick cost.

DER follows pyannote's convention (no collar, overlap included); a 0.25 s
collar variant is reported as well.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/benchmark_baseline.py \
      --data benchmarks/diarization/data/ami \
      --out benchmarks/diarization/results/pre_sync_baseline.json
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
# pylint: disable=missing-function-docstring,invalid-sequence-index

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# pylint: disable=wrong-import-position
from src.shared.utils.speaker_reconciler import (  # noqa: E402
    SpeakerReconciler,
    SpeakerSegment,
)

SAMPLE_RATE = 16000


def load_audio(path: Path) -> np.ndarray:
    samples, rate = sf.read(str(path), dtype="float32")
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{path}: expected {SAMPLE_RATE} Hz, got {rate} Hz")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def load_rttm(path: Path):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=path.stem)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
        annotation[Segment(start, start + duration)] = speaker
    return annotation


def load_uem(path: Path, fallback_end: float):
    from pyannote.core import Segment, Timeline

    timeline = Timeline(uri=path.stem)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 4:
                timeline.add(Segment(float(parts[2]), float(parts[3])))
    if len(timeline) == 0:
        timeline.add(Segment(0.0, fallback_end))
    return timeline


def to_annotation(segments: list[SpeakerSegment], uri: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for seg in segments:
        if seg.end > seg.start:
            annotation[
                Segment(seg.start, seg.end), f"{seg.start}-{seg.end}"
            ] = seg.speaker
    return annotation.support()


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _normalise(components: dict, value: float) -> dict:
    """DER components as fractions of reference speech; JER as-is."""
    out = {"value": round(float(value), 4)}
    if "total" in components:
        total = float(components["total"]) or 1.0
        out["reference_speech_sec"] = round(total, 2)
        for key in ("missed detection", "false alarm", "confusion"):
            if key in components:
                out[key.replace(" ", "_")] = round(
                    float(components[key]) / total, 4
                )
    return out


def score(metric, reference, hypothesis, uem) -> dict:
    components = metric(reference, hypothesis, uem=uem, detailed=True)
    return _normalise(components, metric.compute_metric(components))


def build_metrics():
    from pyannote.metrics.diarization import (
        DiarizationErrorRate,
        JaccardErrorRate,
    )

    return {
        "der": DiarizationErrorRate(collar=0.0, skip_overlap=False),
        "der_collar_0.25": DiarizationErrorRate(
            collar=0.25, skip_overlap=False
        ),
        "jer": JaccardErrorRate(collar=0.0, skip_overlap=False),
    }


def run_offline(service, samples: np.ndarray) -> tuple[list, float]:
    start = time.perf_counter()
    segments = service.diarize(samples, SAMPLE_RATE)
    return segments, time.perf_counter() - start


def run_streaming(
    service,
    samples: np.ndarray,
    tick_sec: float,
    max_buffer_sec: float,
    stream_sec: float | None,
    reference,
):
    """
    Replay the job loop: each tick re-diarizes the rolling buffer, offsets
    the result to session time and reconciles labels.

    Returns the hypothesis a viewer sees, plus tick cost / stability stats.
    """
    reconciler = SpeakerReconciler()
    tick = int(tick_sec * SAMPLE_RATE)
    max_buf = int(max_buffer_sec * SAMPLE_RATE)
    limit = (
        len(samples)
        if stream_sec is None
        else min(len(samples), int(stream_sec * SAMPLE_RATE))
    )

    tick_costs: list[float] = []
    # Label a viewer sees for the newest tick (what in-progress words get)
    first_seen: list[SpeakerSegment] = []
    # Label each region settles on once it has been re-diarized a couple of
    # times (approximates what finalized words get after LocalAgree)
    settled_by_region: dict[int, list[SpeakerSegment]] = {}
    region_label_history: dict[int, list[str | None]] = {}
    onset_latency: list[float] = []

    onsets = [seg.start for seg, _ in reference.itertracks()]
    pending_onsets = sorted(o for o in onsets if o < limit / SAMPLE_RATE)

    for end in range(tick, limit + 1, tick):
        window_start = max(0, end - max_buf)
        window = samples[window_start:end]
        offset_sec = window_start / SAMPLE_RATE

        started = time.perf_counter()
        raw = service.diarize(window, SAMPLE_RATE)
        cost = time.perf_counter() - started
        tick_costs.append(cost)

        session_relative = [
            SpeakerSegment(
                start=seg.start + offset_sec,
                end=seg.end + offset_sec,
                speaker=seg.speaker,
            )
            for seg in raw
        ]
        reconciled = reconciler.reconcile(session_relative)

        tick_end_sec = end / SAMPLE_RATE
        newest_start = tick_end_sec - tick_sec
        first_seen.extend(_clip(reconciled, newest_start, tick_end_sec))

        # Track the dominant label for every tick-sized region in the buffer
        region_first = int(round(window_start / tick))
        region_last = int(round(end / tick))
        for region in range(region_first, region_last):
            r_start, r_end = region * tick_sec, (region + 1) * tick_sec
            clipped = _clip(reconciled, r_start, r_end)
            dominant = _dominant_label(clipped)
            region_label_history.setdefault(region, []).append(dominant)
            # "settled" = label seen two ticks after the region arrived
            if len(region_label_history[region]) == 3:
                settled_by_region[region] = clipped

        # Label latency: time from reference speech onset until the first
        # tick whose hypothesis covers it, including that tick's compute cost
        still_pending = []
        for onset in pending_onsets:
            if onset > tick_end_sec:
                still_pending.append(onset)
                continue
            covered = any(
                seg.start <= onset + 0.25 <= seg.end for seg in reconciled
            )
            if covered:
                onset_latency.append((tick_end_sec - onset) + cost)
            else:
                still_pending.append(onset)
        pending_onsets = still_pending

    # Regions near the end that never reached three ticks keep the label
    # they were first seen with
    for region in region_label_history:
        if region not in settled_by_region:
            r_start, r_end = region * tick_sec, (region + 1) * tick_sec
            settled_by_region[region] = _clip(first_seen, r_start, r_end)
    settled = [seg for segs in settled_by_region.values() for seg in segs]

    flips = 0
    for history in region_label_history.values():
        flips += sum(
            1 for a, b in zip(history, history[1:]) if a != b and b is not None
        )
    minutes = limit / SAMPLE_RATE / 60.0

    return {
        "first_seen": first_seen,
        "settled": settled,
        "streamed_sec": round(limit / SAMPLE_RATE, 1),
        "ticks": len(tick_costs),
        "tick_cost_sec": {
            "mean": round(float(np.mean(tick_costs)), 3),
            "p95": round(float(np.percentile(tick_costs, 95)), 3),
            "worst": round(float(np.max(tick_costs)), 3),
            "budget": tick_sec,
            "fits_budget": bool(np.max(tick_costs) < tick_sec),
        },
        "label_latency_sec": {
            "mean": (
                round(float(np.mean(onset_latency)), 2)
                if onset_latency
                else None
            ),
            "p95": (
                round(float(np.percentile(onset_latency, 95)), 2)
                if onset_latency
                else None
            ),
            "onsets_scored": len(onset_latency),
            "onsets_never_labelled": len(pending_onsets),
        },
        "label_flips_per_min": round(flips / minutes, 2) if minutes else None,
        "session_labels_minted": reconciler._next_label_id,  # pylint: disable=protected-access
    }


def _clip(segments, start: float, end: float) -> list[SpeakerSegment]:
    out = []
    for seg in segments:
        s, e = max(seg.start, start), min(seg.end, end)
        if e > s:
            out.append(SpeakerSegment(start=s, end=e, speaker=seg.speaker))
    return out


def _dominant_label(segments) -> str | None:
    totals: dict[str, float] = {}
    for seg in segments:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + (
            seg.end - seg.start
        )
    return max(totals, key=totals.get) if totals else None


def git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="folder of WAV+RTTM")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--suffix", default="_10min", help="file stem suffix")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tick-sec", type=float, default=5.0)
    parser.add_argument("--max-buffer-sec", type=float, default=30.0)
    parser.add_argument(
        "--stream-sec",
        type=float,
        default=None,
        help="only replay the first N seconds in streaming mode (cost control)",
    )
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--label", default="", help="free-text run label")
    args = parser.parse_args()

    import logging

    import torch

    from src.shared.logger import ContextLogger
    from src.transcription_contexts.pyannote_diarization_context import (
        PyannoteDiarizationContext,
    )

    token_var = "HUGGINGFACE_ACCESS_TOKEN"
    if not os.environ.get(token_var) and os.environ.get("HF_TOKEN"):
        os.environ[token_var] = os.environ["HF_TOKEN"]

    data = Path(args.data)
    wavs = sorted(data.glob(f"*{args.suffix}.wav"))
    if not wavs:
        raise SystemExit(f"No *{args.suffix}.wav files in {data}")

    torch.set_num_threads(os.cpu_count() or 1)
    load_start = time.perf_counter()
    context = PyannoteDiarizationContext(
        {"device": args.device, "token_env_var": token_var}, ["benchmark"]
    )
    logging.basicConfig(level=logging.INFO)
    service = context.create(ContextLogger(logging.getLogger("benchmark")))
    load_sec = time.perf_counter() - load_start
    service.diarize(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE)

    offline_metrics = build_metrics()
    first_seen_metrics = build_metrics()
    settled_metrics = build_metrics()
    files = []

    for wav in wavs:
        stem = wav.stem
        print(f"\n### {stem}", flush=True)
        samples = load_audio(wav)
        audio_sec = len(samples) / SAMPLE_RATE
        reference = load_rttm(wav.with_suffix(".rttm"))
        uem = load_uem(wav.with_suffix(".uem"), audio_sec)

        segments, wall = run_offline(service, samples)
        hyp = to_annotation(segments, stem)
        entry = {
            "file": wav.name,
            "audio_sec": round(audio_sec, 1),
            "reference_speakers": len(reference.labels()),
            "offline": {
                "hypothesis_speakers": len(hyp.labels()),
                "wall_sec": round(wall, 1),
                "rtf": round(wall / audio_sec, 4),
            }
            | {
                k: score(m, reference, hyp, uem)
                for k, m in offline_metrics.items()
            },
        }
        print(
            f"offline: DER {entry['offline']['der']['value']:.3f}  "
            f"RTF {entry['offline']['rtf']:.3f}",
            flush=True,
        )

        if not args.skip_streaming:
            stream = run_streaming(
                service,
                samples,
                args.tick_sec,
                args.max_buffer_sec,
                args.stream_sec,
                reference,
            )
            from pyannote.core import Segment, Timeline

            stream_uem = uem.crop(
                Timeline([Segment(0.0, stream["streamed_sec"])])
            )
            hyp_first = to_annotation(stream.pop("first_seen"), stem)
            hyp_settled = to_annotation(stream.pop("settled"), stem)
            stream["first_seen"] = {
                k: score(m, reference, hyp_first, stream_uem)
                for k, m in first_seen_metrics.items()
            }
            stream["settled"] = {
                k: score(m, reference, hyp_settled, stream_uem)
                for k, m in settled_metrics.items()
            }
            entry["streaming"] = stream
            print(
                f"streaming: first-seen DER "
                f"{stream['first_seen']['der']['value']:.3f}, settled DER "
                f"{stream['settled']['der']['value']:.3f}, tick worst "
                f"{stream['tick_cost_sec']['worst']:.2f}s",
                flush=True,
            )
        files.append(entry)

    def aggregate(metrics) -> dict:
        return {
            name: _normalise(metric[:], abs(metric))
            for name, metric in metrics.items()
        }

    report = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_revision": git_rev(ROOT),
        "environment": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyannote_audio": __import__("pyannote.audio").audio.__version__,
            "device": args.device,
            "cpu_threads": torch.get_num_threads(),
        },
        "config": {
            "model": context._config.model,  # pylint: disable=protected-access
            "tick_sec": args.tick_sec,
            "max_buffer_sec": args.max_buffer_sec,
            "stream_sec": args.stream_sec,
            "der_convention": "pyannote default: no collar, overlap scored",
        },
        "model_load_sec": round(load_sec, 1),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "aggregate": {
            "offline": aggregate(offline_metrics),
            **(
                {}
                if args.skip_streaming
                else {
                    "streaming_first_seen": aggregate(first_seen_metrics),
                    "streaming_settled": aggregate(settled_metrics),
                }
            ),
        },
        "files": files,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"peak RSS {report['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
