"""
Diarization window / segmentation-step sweep.

Replays the streaming job loop (benchmark_baseline.run_streaming) over the
AMI set for every combination of diarization window length and pyannote
segmentation step, so the Phase 2a defaults are chosen from numbers rather
than guessed. Only the streaming pass runs (no offline pass): the question is
"what does a viewer get per tick and what does the tick cost", not best-case
accuracy.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/window_sweep.py \
      --windows 8 10 12 15 20 --steps 0.1 0.25 0.5 \
      --out benchmarks/diarization/results/window_sweep.json
"""

# pylint: disable=too-many-locals,import-outside-toplevel,protected-access
# pylint: disable=missing-function-docstring

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    ROOT,
    SAMPLE_RATE,
    ensure_hf_token_env,
    environment_info,
    git_dirty,
    git_rev,
    load_audio,
    load_rttm,
    load_uem,
    now_iso,
    rel_path,
    write_json,
)
from benchmark_baseline import (  # noqa: E402
    TimedPipeline,
    build_metrics,
    run_streaming,
    score,
    to_annotation,
)

sys.path.insert(0, str(ROOT))


def set_segmentation_step(pipeline, step_ratio: float) -> float:
    """
    Sets pyannote's segmentation sliding-window step as a ratio of the
    segmentation window (the pipeline's own `segmentation_step` convention,
    default 0.1 = 1 s for the 10 s window). Returns the step in seconds.
    """
    inference = pipeline._segmentation
    duration = float(inference.duration)
    inference.step = step_ratio * duration
    pipeline.segmentation_step = step_ratio
    return inference.step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=str(ROOT / "benchmarks/diarization/data/ami")
    )
    parser.add_argument("--suffix", default="_10min")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument(
        "--windows", nargs="+", type=float, default=[8, 10, 12, 15, 20]
    )
    parser.add_argument(
        "--steps", nargs="+", type=float, default=[0.1, 0.25, 0.5]
    )
    parser.add_argument("--tick-sec", type=float, default=5.0)
    parser.add_argument("--stream-sec", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import torch

    from src.shared.logger import ContextLogger
    from src.transcription_contexts.pyannote_diarization_context import (
        PyannoteDiarizationContext,
    )

    token_var = ensure_hf_token_env()
    if args.threads:
        threads = args.threads
    elif hasattr(os, "sched_getaffinity"):
        threads = len(os.sched_getaffinity(0))
    else:
        threads = os.cpu_count() or 1
    torch.set_num_threads(int(threads))

    data = Path(args.data)
    if args.files:
        wavs = [data / f"{stem}.wav" for stem in args.files]
    else:
        wavs = sorted(data.glob(f"*{args.suffix}.wav"))
    if not wavs:
        raise SystemExit(f"No *{args.suffix}.wav files in {data}")

    logging.basicConfig(level=logging.WARNING)
    context = PyannoteDiarizationContext(
        {"device": "cpu", "token_env_var": token_var}, ["benchmark"]
    )
    service = context.create(ContextLogger(logging.getLogger("sweep")))
    timed = TimedPipeline(service._pipeline)
    service._pipeline = timed
    service.diarize(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE)

    audio = {}
    for wav in wavs:
        samples = load_audio(wav)
        reference = load_rttm(wav.with_suffix(".rttm"))
        uem = load_uem(wav.with_suffix(".uem"), len(samples) / SAMPLE_RATE)
        audio[wav.stem] = (samples, reference, uem)

    stream_sec = None if not args.stream_sec else args.stream_sec
    configs = []
    for window in args.windows:
        for step in args.steps:
            step_sec = set_segmentation_step(timed._pipeline, step)
            first_seen_metrics = build_metrics()
            settled_metrics = build_metrics()
            files = []
            started = time.perf_counter()
            for stem, (samples, reference, uem) in audio.items():
                stream = run_streaming(
                    service,
                    timed,
                    samples,
                    args.tick_sec,
                    window,
                    stream_sec,
                    reference,
                )
                from pyannote.core import Segment, Timeline

                stream_uem = uem.crop(
                    Timeline([Segment(0.0, stream["streamed_sec"])])
                )
                hypotheses = stream.pop("hypotheses")
                stream.pop("overlap_hypotheses")
                hyp_first = to_annotation(hypotheses["first_seen"], stem)
                hyp_settled = to_annotation(hypotheses["settled"], stem)
                stream["first_seen"] = {
                    k: score(m, reference, hyp_first, stream_uem)
                    for k, m in first_seen_metrics.items()
                }
                stream["settled"] = {
                    k: score(m, reference, hyp_settled, stream_uem)
                    for k, m in settled_metrics.items()
                }
                files.append({"file": stem, "streaming": stream})
            wall = time.perf_counter() - started

            def agg(metrics):
                return {name: round(abs(m), 4) for name, m in metrics.items()}

            tick_costs = [f["streaming"]["tick_cost_sec"] for f in files]
            entry = {
                "window_sec": window,
                "segmentation_step": step,
                "segmentation_step_sec": round(step_sec, 3),
                "first_seen": agg(first_seen_metrics),
                "settled": agg(settled_metrics),
                "tick_cost_mean_sec": round(
                    float(np.mean([t["mean"] for t in tick_costs])), 3
                ),
                "tick_cost_p95_sec": round(
                    max(t["p95"] for t in tick_costs), 3
                ),
                "tick_cost_worst_sec": round(
                    max(t["worst"] for t in tick_costs), 3
                ),
                "stage_cost_sec": timed.stage_summary(),
                "labels_minted_per_reference_speaker": round(
                    float(
                        np.mean(
                            [
                                f["streaming"][
                                    "labels_minted_per_reference_speaker"
                                ]
                                or 0
                                for f in files
                            ]
                        )
                    ),
                    2,
                ),
                "label_latency_p50_sec": round(
                    float(
                        np.mean(
                            [
                                f["streaming"]["label_latency_sec"]["p50"] or 0
                                for f in files
                            ]
                        )
                    ),
                    2,
                ),
                "onsets_never_labelled": sum(
                    f["streaming"]["label_latency_sec"]["onsets_never_labelled"]
                    for f in files
                ),
                "label_flip_rate_after_first_shown": round(
                    float(
                        np.mean(
                            [
                                f["streaming"][
                                    "label_flip_rate_after_first_shown"
                                ]
                                or 0
                                for f in files
                            ]
                        )
                    ),
                    3,
                ),
                "wall_sec": round(wall, 1),
                "files": files,
            }
            configs.append(entry)
            print(
                f"window {window:>4}s step {step:<4} ({step_sec:.1f}s): "
                f"first-seen DER {entry['first_seen']['der']:.3f} settled DER "
                f"{entry['settled']['der']:.3f} tick mean {entry['tick_cost_mean_sec']:.2f}s "
                f"p95 {entry['tick_cost_p95_sec']:.2f}s minted/spk "
                f"{entry['labels_minted_per_reference_speaker']} "
                f"label p50 {entry['label_latency_p50_sec']}s",
                flush=True,
            )

    report = {
        "label": args.label,
        "generated_at": now_iso(),
        "code_revision": git_rev(ROOT),
        "code_dirty": git_dirty(ROOT),
        "environment": environment_info("cpu")
        | {"cpu_threads": torch.get_num_threads()},
        "config": {
            "tick_sec": args.tick_sec,
            "stream_sec": stream_sec,
            "data": rel_path(data),
            "files": [w.stem for w in wavs],
        },
        "configs": configs,
    }
    write_json(Path(args.out), report)
    print(f"\nWrote {args.out}")
    print(
        json.dumps(
            [
                {
                    k: c[k]
                    for k in (
                        "window_sec",
                        "segmentation_step",
                        "tick_cost_mean_sec",
                        "tick_cost_p95_sec",
                        "labels_minted_per_reference_speaker",
                    )
                }
                | {
                    "first_seen_der": c["first_seen"]["der"],
                    "settled_der": c["settled"]["der"],
                }
                for c in configs
            ],
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
