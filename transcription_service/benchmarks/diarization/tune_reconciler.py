"""
Phase 2b threshold tuning for the speaker reconciler.

Pyannote's per-window output (segments and per-speaker centroids) does not
depend on the reconciler, so it is computed once per window setting and
cached; every reconciler configuration of the grid is then replayed over
the cache through the real `SpeakerReconciler` and `SpeakerLabelAttacher`
(the production path) and scored on the whole set. Report: one row per
configuration with first-seen and settled DER, confusion, labels minted
per reference speaker, speaker-count error and revisions, so the tradeoff
is visible rather than one file's optimum.

Usage (from transcription_service/):
    uv run python benchmarks/diarization/tune_reconciler.py \\
        --data benchmarks/diarization/data/ami --suffix _10min \\
        --window 10 --cache benchmarks/diarization/results/phase2b/cache_w10.pkl \\
        --grid benchmarks/diarization/configs/tune_grid.json \\
        --out benchmarks/diarization/results/phase2b/tune_w10.json
"""

# pylint: disable=too-many-locals,too-many-statements,import-outside-toplevel

import argparse
import itertools
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench_common import (  # noqa: E402
    ensure_hf_token_env,
    git_rev,
    load_audio,
    load_rttm,
    load_uem,
    now_iso,
    rel_path,
    summarize,
    write_json,
)
from benchmark_baseline import (  # noqa: E402
    SAMPLE_RATE,
    _normalise,
    build_metrics,
    score,
    speakers_in,
    to_annotation,
)

from src.shared.utils.speaker_attribution import (  # noqa: E402
    SpeakerLabelAttacher,
)
from src.shared.utils.speaker_reconciler import (  # noqa: E402
    SpeakerReconciler,
    SpeakerReconcilerConfig,
    SpeakerSegment,
)

RECONCILER_KEYS = {
    "match_threshold",
    "new_speaker_threshold",
    "attach_threshold",
    "min_mint_duration_sec",
    "max_speakers",
    "merge_threshold",
    "overlap_bonus",
    "min_update_sec",
    "centroid_memory_sec",
    "max_candidates",
    "min_mint_passes",
}
ATTACHER_KEYS = {"revision_margin", "edge_margin_sec"}


def build_cache(args, wavs: list[Path]) -> dict:
    """
    Runs pyannote once per tick over every file and stores what the
    reconciler will see: session-relative segments (exclusive and
    overlap-aware), embeddings and the tick cost
    """
    import torch

    from src.shared.logger import ContextLogger
    from src.transcription_contexts.pyannote_diarization_context import (
        PyannoteDiarizationContext,
    )

    token_var = ensure_hf_token_env()
    torch.set_num_threads(int(args.threads))
    context = PyannoteDiarizationContext(
        {
            "device": "cpu",
            "token_env_var": token_var,
            "segmentation_step": args.step,
            "clustering_threshold": args.clustering_threshold,
            "local_speakers": not args.no_local_speakers,
            "num_threads": None,
            "nice": 0,
        },
        ["tune"],
    )
    logging.basicConfig(level=logging.WARNING)
    service = context.create(ContextLogger(logging.getLogger("tune")))
    service.diarize(np.zeros(SAMPLE_RATE * 5, dtype=np.float32), SAMPLE_RATE)

    tick = int(args.tick_sec * SAMPLE_RATE)
    window_samples = int(args.window * SAMPLE_RATE)
    cache = {
        "config": {
            "window_sec": args.window,
            "tick_sec": args.tick_sec,
            "segmentation_step": args.step,
            "clustering_threshold": args.clustering_threshold,
            "local_speakers": not args.no_local_speakers,
            "stream_sec": args.stream_sec,
        },
        "files": {},
    }
    for wav in wavs:
        samples = load_audio(wav)
        limit = (
            len(samples)
            if not args.stream_sec
            else min(len(samples), int(args.stream_sec * SAMPLE_RATE))
        )
        passes = []
        started = time.perf_counter()
        for end in range(tick, limit + 1, tick):
            window_start = max(0, end - window_samples)
            offset = window_start / SAMPLE_RATE
            t0 = time.perf_counter()
            result = service.diarize(samples[window_start:end], SAMPLE_RATE)
            cost = time.perf_counter() - t0
            passes.append(
                {
                    "window_start": offset,
                    "window_end": end / SAMPLE_RATE,
                    "segments": [
                        (s.start + offset, s.end + offset, s.speaker)
                        for s in result.segments
                    ],
                    "overlap_segments": [
                        (s.start + offset, s.end + offset, s.speaker)
                        for s in result.overlap_segments
                    ],
                    "embeddings": {
                        k: np.asarray(v, dtype=np.float32)
                        for k, v in result.embeddings.items()
                    },
                    "cost": cost,
                }
            )
        cache["files"][wav.stem] = {
            "wav": str(wav),
            "streamed_sec": limit / SAMPLE_RATE,
            "passes": passes,
        }
        print(
            f"cached {wav.stem}: {len(passes)} passes in "
            f"{time.perf_counter() - started:.0f}s "
            f"(mean tick {np.mean([p['cost'] for p in passes]):.2f}s)",
            flush=True,
        )
    return cache


def replay(
    file_cache: dict,
    reconciler_config: SpeakerReconcilerConfig,
    attacher_kwargs: dict,
    tick_sec: float,
    overlap_aware: bool = False,
) -> dict:
    """
    Replays one file's cached passes through the production reconciler and
    attacher. Returns first-seen, settled and end-of-stream hypotheses plus
    bookkeeping.
    """
    reconciler = SpeakerReconciler(config=reconciler_config)
    # Unbounded history: the replay reads the whole timeline at the end
    attacher = SpeakerLabelAttacher(
        attach_gap_sec=0.0, history_sec=1e9, **attacher_kwargs
    )
    first_seen: list[SpeakerSegment] = []
    settled: list[SpeakerSegment] = []
    settled_through = 0.0
    labels_used: set[str] = set()
    merges = 0
    key = "overlap_segments" if overlap_aware else "segments"
    for pass_ in file_cache["passes"]:
        segments = [SpeakerSegment(*s) for s in pass_[key]]
        before = attacher.covered_through
        reconciled = reconciler.reconcile(segments, pass_["embeddings"])
        labels_used.update(s.speaker for s in reconciled)
        merges += len(reconciler.last_merges)
        attacher.add_coverage(
            reconciled,
            pass_["window_start"],
            pass_["window_end"],
            now=pass_["window_end"],
            confidences=reconciler.last_confidence,
            relabel=dict(reconciler.last_merges),
        )
        after = attacher.covered_through
        first_seen.extend(_timeline(attacher, before, after))
        # Settled: what a region shows two ticks after it arrived
        settle_to = max(settled_through, pass_["window_end"] - 2 * tick_sec)
        settled.extend(_timeline(attacher, settled_through, settle_to))
        settled_through = settle_to
    settled.extend(
        _timeline(attacher, settled_through, attacher.covered_through)
    )
    end_of_stream = _timeline(attacher, 0.0, attacher.covered_through)
    revisions, revised_sec = attacher.revisions
    return {
        "first_seen": first_seen,
        "settled": settled,
        "end_of_stream": end_of_stream,
        "labels_minted": reconciler.labels_minted,
        "labels_used": len(labels_used),
        "merges": merges,
        "revisions": revisions,
        "revised_sec": revised_sec,
    }


def _timeline(attacher: SpeakerLabelAttacher, start: float, end: float):
    out = []
    if end <= start:
        return out
    for span in attacher._segments:  # pylint: disable=protected-access
        s, e = max(span.start, start), min(span.end, end)
        if e > s:
            out.append(SpeakerSegment(s, e, span.speaker))
    return out


def evaluate(
    cache: dict, references: dict, config: dict, tick_sec: float
) -> dict:
    """
    Scores one configuration over every cached file
    """
    reconciler_config = SpeakerReconcilerConfig(
        **{k: v for k, v in config.items() if k in RECONCILER_KEYS}
    )
    attacher_kwargs = {k: v for k, v in config.items() if k in ATTACHER_KEYS}
    overlap_aware = bool(config.get("overlap_aware", False))
    metrics = {
        kind: build_metrics()
        for kind in ("first_seen", "settled", "end_of_stream")
    }
    per_file = {}
    count_errors = []
    minted_ratio = []
    revised = []
    merges = 0
    for stem, file_cache in cache["files"].items():
        reference, uem = references[stem]
        streamed = file_cache["streamed_sec"]
        from pyannote.core import Segment, Timeline

        stream_uem = uem.crop(Timeline([Segment(0.0, streamed)]))
        result = replay(
            file_cache,
            reconciler_config,
            attacher_kwargs,
            tick_sec,
            overlap_aware,
        )
        entry = {}
        for kind in metrics:
            hyp = to_annotation(result[kind], stem)
            entry[kind] = {
                k: score(m, reference, hyp, stream_uem)
                for k, m in metrics[kind].items()
            }
        ref_speakers = speakers_in(reference, 0.0, streamed)
        entry["reference_speakers"] = ref_speakers
        entry["labels_minted"] = result["labels_minted"]
        entry["labels_used"] = result["labels_used"]
        entry["speaker_count_error"] = result["labels_used"] - ref_speakers
        entry["merges"] = result["merges"]
        entry["revisions"] = result["revisions"]
        entry["revised_sec"] = round(result["revised_sec"], 1)
        per_file[stem] = entry
        count_errors.append(abs(entry["speaker_count_error"]))
        minted_ratio.append(result["labels_minted"] / max(1, ref_speakers))
        revised.append(result["revised_sec"] / max(1.0, streamed))
        merges += result["merges"]

    def agg(kind):
        der = _normalise(metrics[kind]["der"][:], abs(metrics[kind]["der"]))
        jer = abs(metrics[kind]["jer"])
        return {
            "der": der["value"],
            "missed": der.get("missed_detection"),
            "false_alarm": der.get("false_alarm"),
            "confusion": der.get("confusion"),
            "jer": round(float(jer), 4),
        }

    aggregate = {kind: agg(kind) for kind in metrics}
    return {
        "config": config,
        "aggregate": aggregate,
        "speaker_count_abs_error_mean": round(float(np.mean(count_errors)), 3),
        "speaker_count_within_1_fraction": round(
            float(np.mean([e <= 1 for e in count_errors])), 3
        ),
        "labels_minted_per_reference_speaker": round(
            float(np.mean(minted_ratio)), 3
        ),
        "revised_fraction_of_audio": round(float(np.mean(revised)), 4),
        "merges": merges,
        "files": per_file,
    }


def expand_grid(grid: dict) -> list[dict]:
    """
    {"param": [values], ...} -> every combination. A value of the form
    {"relative_to": "match_threshold", "offset": -0.1} is resolved against
    the combination.
    """
    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        for key, value in list(combo.items()):
            if isinstance(value, dict) and "relative_to" in value:
                combo[key] = round(
                    combo[value["relative_to"]] + value.get("offset", 0.0), 3
                )
        combos.append(combo)
    return combos


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--suffix", default="_10min")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--clustering-threshold", type=float, default=None)
    parser.add_argument(
        "--no-local-speakers",
        action="store_true",
        help="use the pipeline's clustered output on one-chunk windows "
        "instead of the segmentation's local speaker tracks",
    )
    parser.add_argument("--tick-sec", type=float, default=5.0)
    parser.add_argument("--stream-sec", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache", required=True, help="pickle of raw passes")
    parser.add_argument(
        "--grid", default=None, help="JSON grid; omit to only build the cache"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--score-sec",
        type=float,
        default=0.0,
        help="score only the first N seconds of every file (0 = all cached)",
    )
    parser.add_argument(
        "--sort",
        default="settled",
        choices=["settled", "first_seen", "end_of_stream"],
        help="which DER the printed ranking sorts by",
    )
    args = parser.parse_args()

    data = Path(args.data)
    wavs = (
        [data / f"{stem}.wav" for stem in args.files]
        if args.files
        else sorted(data.glob(f"*{args.suffix}.wav"))
    )
    cache_path = Path(args.cache)
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            cache = pickle.load(handle)
        print(f"loaded cache {rel_path(cache_path)}: {cache['config']}")
    else:
        cache = build_cache(args, wavs)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(cache, handle)
        print(f"wrote cache {rel_path(cache_path)}")
    if not args.grid:
        return

    references = {}
    for stem, file_cache in cache["files"].items():
        wav = Path(file_cache["wav"])
        if args.score_sec:
            file_cache["passes"] = [
                p
                for p in file_cache["passes"]
                if p["window_end"] <= args.score_sec + 1e-6
            ]
            file_cache["streamed_sec"] = min(
                file_cache["streamed_sec"], args.score_sec
            )
        samples_sec = file_cache["streamed_sec"]
        references[stem] = (
            load_rttm(wav.with_suffix(".rttm")),
            load_uem(wav.with_suffix(".uem"), samples_sec),
        )

    with open(args.grid, encoding="utf-8") as handle:
        grid = json.load(handle)
    combos = expand_grid(grid.get("grid", grid))
    print(f"{len(combos)} configurations over {len(cache['files'])} files")
    rows = []
    started = time.perf_counter()
    for index, combo in enumerate(combos):
        rows.append(
            evaluate(cache, references, combo, cache["config"]["tick_sec"])
        )
        if (index + 1) % 25 == 0:
            print(
                f"  {index + 1}/{len(combos)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    rows.sort(key=lambda r: r["aggregate"][args.sort]["der"])

    header = (
        f"{'first':>6} {'settl':>6} {'end':>6} {'conf':>6} {'miss':>6} "
        f"{'fa':>6} {'mint/ref':>8} {'cnt±1':>6} {'rev%':>6} {'merg':>5}  config"
    )
    print(header)
    for row in rows[: args.top]:
        agg = row["aggregate"]
        print(
            f"{agg['first_seen']['der']:6.3f} {agg['settled']['der']:6.3f} "
            f"{agg['end_of_stream']['der']:6.3f} {agg['settled']['confusion']:6.3f} "
            f"{agg['settled']['missed']:6.3f} {agg['settled']['false_alarm']:6.3f} "
            f"{row['labels_minted_per_reference_speaker']:8.2f} "
            f"{row['speaker_count_within_1_fraction']:6.2f} "
            f"{100 * row['revised_fraction_of_audio']:6.2f} {row['merges']:5d}  "
            f"{json.dumps(row['config'])}"
        )
    if args.out:
        write_json(
            Path(args.out),
            {
                "generated_at": now_iso(),
                "code_revision": git_rev(),
                "cache": rel_path(cache_path),
                "cache_config": cache["config"],
                "grid": grid,
                "rows": rows,
            },
        )
        print(f"wrote {rel_path(Path(args.out))}")


if __name__ == "__main__":
    main()
