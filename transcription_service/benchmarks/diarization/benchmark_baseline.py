"""
Accuracy + speed benchmark for ScribeAR's pyannote speaker diarization.

Runs against a folder of 16 kHz mono WAV files with matching reference
RTTM (and optional UEM) files, e.g. AMI meetings prepared by
prepare_ami_baseline.sh or the hard cases from prepare_hard_cases.py, and
writes a JSON report.

Two measurements per file:
  1. offline   - one pipeline pass over the whole file. Best-case DER / JER,
                 speaker-count error, real-time factor, per-stage timing.
  2. streaming - replays the file the way WhisperStreamingProviderJob sees
                 it: every job tick (5 s) the rolling buffer (max 30 s) is
                 re-diarized and raw labels are passed through
                 SpeakerReconciler. Scores the labels a viewer would actually
                 see, plus label latency, label flips, labels minted per real
                 speaker, per-tick cost and per-stage timing, modelled lag
                 behind real time and skipped ticks, and memory growth.

DER follows pyannote's convention (no collar, overlap included); a 0.25 s
collar variant is reported as well.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/benchmark_baseline.py \
      --data benchmarks/diarization/data/ami --suffix _10min \
      --stream-sec 120 --out benchmarks/diarization/results/replay.json

`--stream-sec 0` replays each file in full.
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel
# pylint: disable=missing-function-docstring,invalid-sequence-index
# pylint: disable=too-many-branches,protected-access

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
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
    percentile,
    rel_path,
    summarize,
    write_json,
)

sys.path.insert(0, str(ROOT))
from src.shared.utils.speaker_reconciler import (  # noqa: E402
    SpeakerReconciler,
    SpeakerSegment,
)


class TimedPipeline:
    """
    Wraps the pyannote pipeline held by PyannoteDiarizationService so every
    call also records pyannote's own per-step timing (segmentation,
    embeddings, clustering, ...). Benchmark-side instrumentation only: the
    service code is untouched and the pipeline sees the same input.
    """

    def __init__(self, pipeline):
        self._pipeline = pipeline
        self.stage_times: dict[str, list[float]] = defaultdict(list)
        self.last: dict[str, float] = {}

    def __call__(self, file, **kwargs):
        from pyannote.audio.pipelines.utils.hook import TimingHook

        hook = TimingHook()
        hook.__enter__()
        try:
            output = self._pipeline(file, hook=hook, **kwargs)
        finally:
            try:
                hook.__exit__(None, None, None)
            except AttributeError:
                # No step reported (nothing to time), leave `last` empty
                file["timing"] = {}
        self.last = {
            k: float(v) for k, v in dict(file.get("timing", {})).items()
        }
        # pyannote only hooks the two model stages; what remains of the total
        # is clustering (VBx / agglomerative) and bookkeeping.
        if "total" in self.last:
            known = sum(v for k, v in self.last.items() if k != "total")
            self.last["clustering_and_other"] = max(
                0.0, self.last["total"] - known
            )
        for name, seconds in self.last.items():
            self.stage_times[name].append(seconds)
        return output

    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    def reset(self):
        self.stage_times = defaultdict(list)

    def stage_summary(self) -> dict:
        return {
            name: summarize(values) for name, values in self.stage_times.items()
        }


def to_annotation(segments: list[SpeakerSegment], uri: str):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for seg in segments:
        if seg.end > seg.start:
            annotation[
                Segment(seg.start, seg.end), f"{seg.start}-{seg.end}"
            ] = seg.speaker
    return annotation.support()


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


def speakers_in(reference, start: float, end: float) -> int:
    """Reference speakers with any speech inside [start, end)."""
    from pyannote.core import Segment

    cropped = reference.crop(Segment(start, end), mode="intersection")
    return len(cropped.labels())


def run_offline(service, samples: np.ndarray) -> tuple[list, float]:
    start = time.perf_counter()
    segments = service.diarize(samples, SAMPLE_RATE)
    return segments, time.perf_counter() - start


def model_schedule(tick_costs: list[float], tick_sec: float) -> dict:
    """
    What a real-time worker would have done with these tick costs.

    The worker pool runs one pass per job period and drops every period that
    elapses while the previous pass is still running (worker_process.py:
    dropped_periods += periods_advanced - 1). The replay above runs every
    tick so every second of audio gets scored; this models the schedule
    separately: lag = when a pass finishes minus when its period was due.
    Approximate (a skipped period would have changed the next pass's input)
    but it is the number the live /metrics/status counters correspond to.
    """
    finish = 0.0
    lags: list[float] = []
    skipped = 0
    for index, cost in enumerate(tick_costs):
        due = (index + 1) * tick_sec
        if finish > due:
            skipped += 1
            continue
        finish = max(due, finish) + cost
        lags.append(finish - due)
    return {
        "lag_behind_realtime_sec": summarize(lags),
        "ticks_skipped": skipped,
        "ticks_skipped_fraction": (
            round(skipped / len(tick_costs), 3) if tick_costs else None
        ),
    }


def run_streaming(
    service,
    timed_pipeline: TimedPipeline | None,
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
        if not stream_sec
        else min(len(samples), int(stream_sec * SAMPLE_RATE))
    )

    if timed_pipeline is not None:
        timed_pipeline.reset()
    # Memory: growth is measured from the first streaming tick (steady state
    # with the model loaded) to the last, so an offline pass that ran just
    # before cannot show up as negative "growth" while its memory is freed.
    rss_start: float | None = None
    rss_min = float("inf")
    rss_max = 0.0
    tick_costs: list[float] = []
    reconciler_costs: list[float] = []
    # Label a viewer sees for the newest tick (what in-progress words get)
    first_seen: list[SpeakerSegment] = []
    # Label each region settles on once it has been re-diarized a couple of
    # times (approximates what finalized words get after LocalAgree)
    settled_by_region: dict[int, list[SpeakerSegment]] = {}
    region_label_history: dict[int, list[str | None]] = {}
    onset_latency: list[float] = []
    labels_used: set[str] = set()

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
        rss_now = current_rss_mb()
        if rss_start is None:
            rss_start = rss_now
        rss_min = min(rss_min, rss_now)
        rss_max = max(rss_max, rss_now)

        session_relative = [
            SpeakerSegment(
                start=seg.start + offset_sec,
                end=seg.end + offset_sec,
                speaker=seg.speaker,
            )
            for seg in raw
        ]
        started = time.perf_counter()
        reconciled = reconciler.reconcile(session_relative)
        reconciler_costs.append(time.perf_counter() - started)
        labels_used.update(seg.speaker for seg in reconciled)

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

    rss_end = current_rss_mb()
    if rss_start is None:
        rss_start = rss_end
        rss_min = rss_max = rss_end

    # Regions near the end that never reached three ticks keep the label
    # they were first seen with
    for region in region_label_history:
        if region not in settled_by_region:
            r_start, r_end = region * tick_sec, (region + 1) * tick_sec
            settled_by_region[region] = _clip(first_seen, r_start, r_end)
    settled = [seg for segs in settled_by_region.values() for seg in segs]

    # Flips: a region's dominant label changing between consecutive ticks
    # (rate per minute, the original metric) and, per region, whether the
    # label ever changed after it was first shown (what a viewer notices).
    flips = 0
    regions_labelled = 0
    regions_flipped = 0
    for history in region_label_history.values():
        flips += sum(
            1 for a, b in zip(history, history[1:]) if a != b and b is not None
        )
        shown = [label for label in history if label is not None]
        if shown:
            regions_labelled += 1
            if any(label != shown[0] for label in shown[1:]):
                regions_flipped += 1
    minutes = limit / SAMPLE_RATE / 60.0
    streamed_sec = limit / SAMPLE_RATE
    reference_speakers_streamed = speakers_in(reference, 0.0, streamed_sec)

    return {
        "first_seen": first_seen,
        "settled": settled,
        "streamed_sec": round(streamed_sec, 1),
        "ticks": len(tick_costs),
        "tick_cost_sec": {
            **summarize(tick_costs),
            "worst": (
                round(float(np.max(tick_costs)), 3) if tick_costs else None
            ),
            "budget": tick_sec,
            "fits_budget": bool(tick_costs and np.max(tick_costs) < tick_sec),
        },
        "reconciler_cost_sec": summarize(reconciler_costs, digits=6),
        "stage_cost_sec": (
            timed_pipeline.stage_summary() if timed_pipeline else {}
        ),
        "schedule": model_schedule(tick_costs, tick_sec),
        "label_latency_sec": {
            "mean": (
                round(float(np.mean(onset_latency)), 2)
                if onset_latency
                else None
            ),
            "p50": (
                round(percentile(onset_latency, 0.5), 2)
                if onset_latency
                else None
            ),
            "p95": (
                round(percentile(onset_latency, 0.95), 2)
                if onset_latency
                else None
            ),
            "onsets_scored": len(onset_latency),
            "onsets_never_labelled": len(pending_onsets),
        },
        "label_flips_per_min": round(flips / minutes, 2) if minutes else None,
        "label_flip_rate_after_first_shown": (
            round(regions_flipped / regions_labelled, 3)
            if regions_labelled
            else None
        ),
        "regions_labelled": regions_labelled,
        "session_labels_minted": reconciler.labels_minted,
        "session_labels_used": len(labels_used),
        "reference_speakers_streamed": reference_speakers_streamed,
        "labels_minted_per_reference_speaker": (
            round(reconciler.labels_minted / reference_speakers_streamed, 2)
            if reference_speakers_streamed
            else None
        ),
        "speaker_count_error": len(labels_used) - reference_speakers_streamed,
        "memory": {
            "rss_after_first_tick_mb": round(rss_start, 1),
            "rss_end_mb": round(rss_end, 1),
            "rss_min_mb": round(rss_min, 1),
            "rss_max_mb": round(rss_max, 1),
            "growth_mb": round(rss_end - rss_start, 1),
        },
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


def _mean_of(files: list[dict], path: list[str], digits: int = 3):
    values = []
    for entry in files:
        node = entry
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            values.append(float(node))
    return round(float(np.mean(values)), digits) if values else None


def _max_of(files: list[dict], path: list[str], digits: int = 3):
    values = []
    for entry in files:
        node = entry
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            values.append(float(node))
    return round(max(values), digits) if values else None


def _pool_stages(files: list[dict]) -> dict:
    """Weighted mean per pyannote stage across files (by call count)."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    worst: dict[str, float] = defaultdict(float)
    for entry in files:
        stages = entry.get("streaming", {}).get("stage_cost_sec", {})
        for name, stat in stages.items():
            if stat.get("mean") is not None:
                totals[name][0] += stat["mean"] * stat["count"]
                totals[name][1] += stat["count"]
                worst[name] = max(worst[name], stat["max"] or 0.0)
    return {
        name: {
            "mean": round(total / count, 3) if count else None,
            "max": round(worst[name], 3),
            "count": count,
        }
        for name, (total, count) in totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="folder of WAV+RTTM")
    parser.add_argument("--out", required=True, help="JSON report path")
    parser.add_argument("--suffix", default="_10min", help="file stem suffix")
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="explicit WAV stems to run (default: every *<suffix>.wav)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tick-sec", type=float, default=5.0)
    parser.add_argument("--max-buffer-sec", type=float, default=30.0)
    parser.add_argument(
        "--stream-sec",
        type=float,
        default=120.0,
        help="replay only the first N seconds of each file (0 = full length)",
    )
    parser.add_argument("--skip-streaming", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch threads (default: all CPUs the process may use)",
    )
    parser.add_argument("--label", default="", help="free-text run label")
    parser.add_argument(
        "--no-hygiene",
        action="store_true",
        help="skip the hygiene snapshot (the suite runner records its own)",
    )
    args = parser.parse_args()

    import torch

    from src.shared.logger import ContextLogger
    from src.transcription_contexts.pyannote_diarization_context import (
        PyannoteDiarizationContext,
    )

    token_var = ensure_hf_token_env()
    hygiene = None if args.no_hygiene else hygiene_check()

    data = Path(args.data)
    if args.files:
        wavs = [data / f"{stem}.wav" for stem in args.files]
        missing = [w for w in wavs if not w.exists()]
        if missing:
            raise SystemExit(f"missing: {missing}")
    else:
        wavs = sorted(data.glob(f"*{args.suffix}.wav"))
    if not wavs:
        raise SystemExit(f"No *{args.suffix}.wav files in {data}")

    if args.threads:
        threads = args.threads
    elif hasattr(os, "sched_getaffinity"):
        threads = len(os.sched_getaffinity(0))
    else:
        threads = os.cpu_count() or 1
    torch.set_num_threads(int(threads))

    # Warm-up: load the model once and run a short pass so the first timed
    # call does not pay for lazy kernel initialisation or the download.
    load_start = time.perf_counter()
    context = PyannoteDiarizationContext(
        {"device": args.device, "token_env_var": token_var}, ["benchmark"]
    )
    logging.basicConfig(level=logging.INFO)
    service = context.create(ContextLogger(logging.getLogger("benchmark")))
    load_sec = time.perf_counter() - load_start
    timed = TimedPipeline(service._pipeline)
    service._pipeline = timed
    warm_start = time.perf_counter()
    service.diarize(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE)
    warm_sec = time.perf_counter() - warm_start

    offline_metrics = build_metrics()
    first_seen_metrics = build_metrics()
    settled_metrics = build_metrics()
    files = []
    stream_sec = None if not args.stream_sec else args.stream_sec

    for wav in wavs:
        stem = wav.stem
        print(f"\n### {stem}", flush=True)
        samples = load_audio(wav)
        audio_sec = len(samples) / SAMPLE_RATE
        reference = load_rttm(wav.with_suffix(".rttm"))
        uem = load_uem(wav.with_suffix(".uem"), audio_sec)
        entry: dict = {
            "file": wav.name,
            "audio_sec": round(audio_sec, 1),
            "reference_speakers": len(reference.labels()),
        }

        if not args.skip_offline:
            timed.reset()
            segments, wall = run_offline(service, samples)
            hyp = to_annotation(segments, stem)
            entry["offline"] = {
                "hypothesis_speakers": len(hyp.labels()),
                "speaker_count_error": len(hyp.labels())
                - len(reference.labels()),
                "wall_sec": round(wall, 1),
                "rtf": round(wall / audio_sec, 4),
                "stage_cost_sec": {
                    k: round(v, 3) for k, v in timed.last.items()
                },
            } | {
                k: score(m, reference, hyp, uem)
                for k, m in offline_metrics.items()
            }
            print(
                f"offline: DER {entry['offline']['der']['value']:.3f}  "
                f"RTF {entry['offline']['rtf']:.3f}  speakers "
                f"{entry['offline']['hypothesis_speakers']}/"
                f"{entry['reference_speakers']}",
                flush=True,
            )

        if not args.skip_streaming:
            stream = run_streaming(
                service,
                timed,
                samples,
                args.tick_sec,
                args.max_buffer_sec,
                stream_sec,
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
                f"{stream['settled']['der']['value']:.3f}, tick p95 "
                f"{stream['tick_cost_sec']['p95']:.2f}s worst "
                f"{stream['tick_cost_sec']['worst']:.2f}s, labels "
                f"{stream['session_labels_minted']} for "
                f"{stream['reference_speakers_streamed']} speakers, "
                f"label latency p50 {stream['label_latency_sec']['p50']}s",
                flush=True,
            )
        files.append(entry)

    def aggregate(metrics) -> dict:
        return {
            name: _normalise(metric[:], abs(metric))
            for name, metric in metrics.items()
        }

    aggregate_report: dict = {}
    if not args.skip_offline:
        aggregate_report["offline"] = aggregate(offline_metrics) | {
            "speaker_count_abs_error_mean": _mean_of(
                [
                    {"v": abs(f["offline"]["speaker_count_error"])}
                    for f in files
                    if "offline" in f
                ],
                ["v"],
            ),
            "rtf_mean": _mean_of(files, ["offline", "rtf"], 4),
        }
    if not args.skip_streaming:
        aggregate_report["streaming_first_seen"] = aggregate(first_seen_metrics)
        aggregate_report["streaming_settled"] = aggregate(settled_metrics)
        aggregate_report["streaming"] = {
            "tick_cost_mean_sec": _mean_of(
                files, ["streaming", "tick_cost_sec", "mean"]
            ),
            "tick_cost_p95_sec": _max_of(
                files, ["streaming", "tick_cost_sec", "p95"]
            ),
            "tick_cost_worst_sec": _max_of(
                files, ["streaming", "tick_cost_sec", "worst"]
            ),
            "lag_behind_realtime_p50_sec": _mean_of(
                files,
                ["streaming", "schedule", "lag_behind_realtime_sec", "p50"],
            ),
            "lag_behind_realtime_p95_sec": _max_of(
                files,
                ["streaming", "schedule", "lag_behind_realtime_sec", "p95"],
            ),
            "ticks_skipped_fraction": _mean_of(
                files, ["streaming", "schedule", "ticks_skipped_fraction"]
            ),
            "label_latency_p50_sec": _mean_of(
                files, ["streaming", "label_latency_sec", "p50"], 2
            ),
            "label_latency_p95_sec": _max_of(
                files, ["streaming", "label_latency_sec", "p95"], 2
            ),
            "onsets_never_labelled": sum(
                f["streaming"]["label_latency_sec"]["onsets_never_labelled"]
                for f in files
                if "streaming" in f
            ),
            "label_flips_per_min": _mean_of(
                files, ["streaming", "label_flips_per_min"], 2
            ),
            "label_flip_rate_after_first_shown": _mean_of(
                files, ["streaming", "label_flip_rate_after_first_shown"]
            ),
            "labels_minted_per_reference_speaker": _mean_of(
                files, ["streaming", "labels_minted_per_reference_speaker"], 2
            ),
            "speaker_count_abs_error_mean": _mean_of(
                [
                    {"v": abs(f["streaming"]["speaker_count_error"])}
                    for f in files
                    if "streaming" in f
                ],
                ["v"],
            ),
            "stage_cost_sec": _pool_stages(files),
            "reconciler_cost_mean_sec": _mean_of(
                files, ["streaming", "reconciler_cost_sec", "mean"], 6
            ),
            "memory_growth_mb_total": round(
                sum(
                    f["streaming"]["memory"]["growth_mb"]
                    for f in files
                    if "streaming" in f
                ),
                1,
            ),
        }

    report = {
        "label": args.label,
        "generated_at": now_iso(),
        "code_revision": git_rev(ROOT),
        "code_dirty": git_dirty(ROOT),
        "environment": environment_info(args.device)
        | {"cpu_threads": torch.get_num_threads()},
        "hygiene": hygiene,
        "config": {
            "model": context._config.model,
            "tick_sec": args.tick_sec,
            "max_buffer_sec": args.max_buffer_sec,
            "stream_sec": stream_sec,
            "der_convention": "pyannote default: no collar, overlap scored",
            "data": rel_path(data),
            "suffix": args.suffix,
        },
        "model_load_sec": round(load_sec, 1),
        "warmup_pass_sec": round(warm_sec, 2),
        "peak_rss_mb": round(peak_rss_mb(), 0),
        "aggregate": aggregate_report,
        "files": files,
    }

    out = Path(args.out)
    write_json(out, report)
    print(f"\nWrote {out}")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"peak RSS {report['peak_rss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
