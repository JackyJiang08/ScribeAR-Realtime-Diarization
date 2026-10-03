"""
Long-session soak harness: replays the concatenated AMI soak stream
(prepare_soak.py) through the streaming job loop for an hour (or two) and
tracks, per time bin, what a long classroom session would show:

  - DER / confusion drift (first-seen labels) per bin
  - label swaps: for each reference speaker, the session label that covers
    most of their speech in the bin; a change from the previous bin is a
    swap the audience would notice
  - session labels minted so far (should stay near the number of people)
  - tick cost per bin and the modelled lag behind real time
  - process RSS per bin and its growth from the first to the last bin

The replay runs at compute speed, not real time: with today's 12 s ticks an
hour of audio takes about 2.5 hours; once the tick fits the 5 s period it
takes roughly RTF x 60 min. Use --stream-min for a shorter check.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/prepare_soak.py --minutes 60
  uv run python benchmarks/diarization/soak.py --minutes 60 \
      --out benchmarks/diarization/results/soak_60min.json
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
# pylint: disable=missing-function-docstring,protected-access

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    DATA_DIR,
    ROOT,
    SAMPLE_RATE,
    current_rss_mb,
    ensure_hf_token_env,
    environment_info,
    git_dirty,
    git_rev,
    hygiene_check,
    load_audio,
    load_rttm,
    load_uem,
    now_iso,
    peak_rss_mb,
    summarize,
    write_json,
)
from benchmark_baseline import (  # noqa: E402
    TimedPipeline,
    _clip,
    build_metrics,
    model_schedule,
    score,
    to_annotation,
)

sys.path.insert(0, str(ROOT))
from src.shared.utils.speaker_reconciler import (  # noqa: E402
    SpeakerReconciler,
    SpeakerSegment,
)


def majority_labels(reference, hypothesis, start: float, end: float) -> dict:
    """Per reference speaker, the hypothesis label covering most of their speech in [start, end)."""
    from pyannote.core import Segment

    window = Segment(start, end)
    ref = reference.crop(window, mode="intersection")
    hyp = hypothesis.crop(window, mode="intersection")
    out = {}
    for speaker in ref.labels():
        timeline = ref.label_timeline(speaker)
        covered = hyp.crop(timeline, mode="intersection")
        chart = covered.chart()
        out[speaker] = chart[0][0] if chart else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=60, choices=[60, 120])
    parser.add_argument("--data", default=str(DATA_DIR / "soak"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--bin-min", type=float, default=5.0)
    parser.add_argument("--tick-sec", type=float, default=5.0)
    parser.add_argument("--max-buffer-sec", type=float, default=30.0)
    parser.add_argument(
        "--stream-min",
        type=float,
        default=0.0,
        help="replay only the first N minutes (0 = whole stream)",
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import torch

    from src.shared.logger import ContextLogger
    from src.transcription_contexts.pyannote_diarization_context import (
        PyannoteDiarizationContext,
    )

    token_var = ensure_hf_token_env()
    hygiene = hygiene_check()
    stem = f"soak_{args.minutes}min"
    wav = Path(args.data) / f"{stem}.wav"
    if not wav.exists():
        raise SystemExit(
            f"{wav} missing; run prepare_soak.py --minutes {args.minutes}"
        )
    manifest_path = Path(args.data) / f"{stem}.MANIFEST.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )

    samples = load_audio(wav)
    reference = load_rttm(wav.with_suffix(".rttm"))
    uem = load_uem(wav.with_suffix(".uem"), len(samples) / SAMPLE_RATE)
    limit = len(samples)
    if args.stream_min:
        limit = min(limit, int(args.stream_min * 60 * SAMPLE_RATE))

    torch.set_num_threads(torch.get_num_threads())
    logging.basicConfig(level=logging.WARNING)
    context = PyannoteDiarizationContext(
        {"device": "cpu", "token_env_var": token_var}, ["soak"]
    )
    service = context.create(ContextLogger(logging.getLogger("soak")))
    timed = TimedPipeline(service._pipeline)
    service._pipeline = timed
    service.diarize(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE)

    reconciler = SpeakerReconciler()
    tick = int(args.tick_sec * SAMPLE_RATE)
    max_buf = int(args.max_buffer_sec * SAMPLE_RATE)
    bin_sec = args.bin_min * 60.0
    first_seen: list[SpeakerSegment] = []
    tick_costs: list[float] = []
    bins: list[dict] = []
    current_bin = 0
    bin_costs: list[float] = []
    rss_first = current_rss_mb()
    previous_majority: dict = {}
    swaps_total = 0
    started_wall = time.perf_counter()

    def close_bin(index: int, costs: list[float]):
        nonlocal previous_majority, swaps_total
        from pyannote.core import Segment, Timeline

        b_start, b_end = index * bin_sec, min(
            (index + 1) * bin_sec, limit / SAMPLE_RATE
        )
        hyp = to_annotation(_clip(first_seen, b_start, b_end), stem)
        bin_uem = uem.crop(Timeline([Segment(b_start, b_end)]))
        metrics = build_metrics()
        der = score(metrics["der"], reference, hyp, bin_uem)
        majority = majority_labels(reference, hyp, b_start, b_end)
        swaps = [
            speaker
            for speaker, label in majority.items()
            if label is not None
            and previous_majority.get(speaker) is not None
            and previous_majority[speaker] != label
        ]
        swaps_total += len(swaps)
        previous_majority = {
            k: (v if v is not None else previous_majority.get(k))
            for k, v in {**previous_majority, **majority}.items()
        }
        rss = current_rss_mb()
        entry = {
            "bin": index,
            "start_min": round(b_start / 60, 1),
            "end_min": round(b_end / 60, 1),
            "der": der,
            "majority_label_per_reference_speaker": majority,
            "label_swaps": swaps,
            "labels_minted_so_far": reconciler.labels_minted,
            "tick_cost_sec": summarize(costs),
            "rss_mb": round(rss, 1),
        }
        bins.append(entry)
        print(
            f"bin {index:3d} ({entry['start_min']:.0f}-{entry['end_min']:.0f} min): "
            f"DER {der['value']:.3f} conf {der.get('confusion', 0):.3f} "
            f"labels {reconciler.labels_minted} swaps {len(swaps)} "
            f"tick mean {entry['tick_cost_sec']['mean']}s rss {rss:.0f} MB",
            flush=True,
        )

    for end in range(tick, limit + 1, tick):
        window_start = max(0, end - max_buf)
        offset_sec = window_start / SAMPLE_RATE
        started = time.perf_counter()
        result = service.diarize(samples[window_start:end], SAMPLE_RATE)
        cost = time.perf_counter() - started
        tick_costs.append(cost)
        bin_costs.append(cost)
        reconciled = reconciler.reconcile(
            [
                SpeakerSegment(
                    seg.start + offset_sec, seg.end + offset_sec, seg.speaker
                )
                for seg in result.segments
            ],
            result.embeddings,
        )
        tick_end = end / SAMPLE_RATE
        first_seen.extend(_clip(reconciled, tick_end - args.tick_sec, tick_end))
        if int(tick_end // bin_sec) > current_bin:
            close_bin(current_bin, bin_costs)
            current_bin += 1
            bin_costs = []
    if bin_costs:
        close_bin(current_bin, bin_costs)

    streamed_sec = limit / SAMPLE_RATE
    from pyannote.core import Segment, Timeline

    overall_uem = uem.crop(Timeline([Segment(0.0, streamed_sec)]))
    metrics = build_metrics()
    hyp_all = to_annotation(first_seen, stem)
    overall = {
        k: score(m, reference, hyp_all, overall_uem) for k, m in metrics.items()
    }
    ref_speakers = len(reference.crop(Segment(0.0, streamed_sec)).labels())
    rss_values = [b["rss_mb"] for b in bins]
    der_values = [b["der"]["value"] for b in bins]
    report = {
        "label": args.label,
        "generated_at": now_iso(),
        "code_revision": git_rev(ROOT),
        "code_dirty": git_dirty(ROOT),
        "environment": environment_info("cpu"),
        "hygiene": hygiene,
        "config": {
            "stream": str(wav),
            "manifest": manifest,
            "streamed_min": round(streamed_sec / 60, 1),
            "bin_min": args.bin_min,
            "tick_sec": args.tick_sec,
            "max_buffer_sec": args.max_buffer_sec,
        },
        "overall": overall
        | {
            "reference_speakers": ref_speakers,
            "session_labels_minted": reconciler.labels_minted,
            "labels_minted_per_reference_speaker": (
                round(reconciler.labels_minted / ref_speakers, 2)
                if ref_speakers
                else None
            ),
            "label_swaps_total": swaps_total,
            "label_swaps_per_hour": (
                round(swaps_total / (streamed_sec / 3600), 2)
                if streamed_sec
                else None
            ),
            "der_first_bin": der_values[0] if der_values else None,
            "der_last_bin": der_values[-1] if der_values else None,
            "der_drift": (
                round(der_values[-1] - der_values[0], 4)
                if len(der_values) > 1
                else None
            ),
            "tick_cost_sec": summarize(tick_costs),
            "stage_cost_sec": timed.stage_summary(),
            "schedule": model_schedule(tick_costs, args.tick_sec),
            "memory": {
                "rss_first_mb": round(rss_first, 1),
                "rss_last_mb": rss_values[-1] if rss_values else None,
                "growth_mb": (
                    round(rss_values[-1] - rss_first, 1) if rss_values else None
                ),
                "peak_rss_mb": round(peak_rss_mb(), 0),
            },
            "wall_sec": round(time.perf_counter() - started_wall, 1),
        },
        "bins": bins,
    }
    write_json(Path(args.out), report)
    print(f"\nWrote {args.out}")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
