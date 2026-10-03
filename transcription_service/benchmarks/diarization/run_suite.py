"""
Standard diarization evaluation suite: one command, one JSON report.

Steps, each in its own subprocess so memory and thread state never leak
between them:
  0. record resource limits and run hygiene (free memory, swap, load)
  1. warm-up: load whisper base, Silero and pyannote once (warmup.py)
  2. replay benchmark on the AMI set (benchmark_baseline.py): offline and
     streaming DER/JER, speaker-count error, label latency, flips, labels
     minted, per-stage timing, modelled lag and skipped ticks, memory
  3. end-to-end caption latency with the reference config, diarization ON
     and OFF (caption_latency.py): chunk-id (primary) and word (secondary)
     methods, dropped periods, dropped audio, service RSS
  4. optionally the same with the old VAD-on dev config, labelled
     `secondary_dev_vad` and never gated

The report's `key_metrics` block is the flat view compare_baseline.py gates.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/run_suite.py \
      --out benchmarks/diarization/results/suite.json --label "my change"
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=missing-function-docstring,broad-exception-caught

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    BENCH_DIR,
    CONFIGS_DIR,
    DATA_DIR,
    ROOT,
    ensure_hf_token_env,
    environment_info,
    git_dirty,
    git_rev,
    hygiene_check,
    now_iso,
    rel_path,
    resource_limits,
    system_state,
    write_json,
)

STEPS = ("warmup", "replay", "caption")


def run_step(name: str, argv: list[str], log_dir: Path) -> float:
    """Runs a benchmark script as a subprocess, teeing its output to a log."""
    log_path = log_dir / f"{name}.log"
    print(f"\n==> {name}: {' '.join(argv)}", flush=True)
    started = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, *argv],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            sys.stdout.write(f"    {line}")
            sys.stdout.flush()
        process.wait()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise SystemExit(
            f"step {name} failed with {process.returncode}; see {log_path}"
        )
    print(f"<== {name} done in {elapsed:.0f}s", flush=True)
    return elapsed


def _get(node, *path, default=None):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def caption_key_metrics(prefix: str, report: dict | None) -> dict:
    if not report:
        return {}
    lat = report["latency"]
    counters = report["service_counters_delta"]
    out = {}
    for method, key in (
        ("chunk_id", "chunk_id_method"),
        ("word", "word_method"),
    ):
        for kind, stats in lat[key].items():
            for stat in ("p50", "p95", "mean"):
                out[f"{prefix}.{method}.{kind}.{stat}"] = stats.get(stat)
    out[f"{prefix}.transcript_messages"] = lat["transcript_messages"]
    out[f"{prefix}.final_words"] = lat["final_words"]
    out[f"{prefix}.final_words_labelled_fraction"] = lat[
        "final_words_labelled_fraction"
    ]
    out[f"{prefix}.speaker_labels_seen"] = len(lat["speaker_labels_seen"])
    out[f"{prefix}.label_changes_before_final"] = lat[
        "label_changes_before_final"
    ]
    out[f"{prefix}.dropped_periods"] = counters.get("asrDroppedPeriodsTotal")
    out[f"{prefix}.audio_dropped_buffer_full_sec"] = counters.get(
        "audioDroppedBufferFullSecondsTotal"
    )
    out[f"{prefix}.buffer_overflow_sec"] = counters.get(
        "bufferOverflowSecondsTotal"
    )
    out[f"{prefix}.jobs_completed"] = counters.get("jobsCompletedTotal")
    out[f"{prefix}.rss_peak_mb"] = report["service"]["rss_mb_peak"]
    out[f"{prefix}.rss_growth_mb"] = report["service"][
        "rss_mb_growth_during_stream"
    ]
    out[f"{prefix}.exec_ms_p95"] = _get(
        report, "service_histograms_end", "asrExecutionMs", "p95"
    )
    out[f"{prefix}.exec_ms_max"] = _get(
        report, "service_histograms_end", "asrExecutionMs", "max"
    )
    return out


def replay_key_metrics(report: dict | None) -> dict:
    if not report:
        return {}
    agg = report["aggregate"]
    out = {}
    offline = agg.get("offline")
    if offline:
        out["replay.offline.der"] = offline["der"]["value"]
        out["replay.offline.der_missed"] = offline["der"].get(
            "missed_detection"
        )
        out["replay.offline.der_false_alarm"] = offline["der"].get(
            "false_alarm"
        )
        out["replay.offline.der_confusion"] = offline["der"].get("confusion")
        out["replay.offline.jer"] = offline["jer"]["value"]
        out["replay.offline.speaker_count_abs_error_mean"] = offline[
            "speaker_count_abs_error_mean"
        ]
        out["replay.offline.rtf_mean"] = offline["rtf_mean"]
    for kind in ("first_seen", "settled"):
        block = agg.get(f"streaming_{kind}")
        if block:
            out[f"replay.{kind}.der"] = block["der"]["value"]
            out[f"replay.{kind}.der_missed"] = block["der"].get(
                "missed_detection"
            )
            out[f"replay.{kind}.der_false_alarm"] = block["der"].get(
                "false_alarm"
            )
            out[f"replay.{kind}.der_confusion"] = block["der"].get("confusion")
            out[f"replay.{kind}.jer"] = block["jer"]["value"]
    streaming = agg.get("streaming")
    if streaming:
        for key in (
            "tick_cost_mean_sec",
            "tick_cost_p95_sec",
            "tick_cost_worst_sec",
            "lag_behind_realtime_p50_sec",
            "lag_behind_realtime_p95_sec",
            "ticks_skipped_fraction",
            "label_latency_p50_sec",
            "label_latency_p95_sec",
            "onsets_never_labelled",
            "label_flips_per_min",
            "label_flip_rate_after_first_shown",
            "labels_minted_per_reference_speaker",
            "speaker_count_abs_error_mean",
            "reconciler_cost_mean_sec",
            "memory_growth_mb_total",
        ):
            out[f"replay.{key}"] = streaming.get(key)
        for stage, stats in streaming.get("stage_cost_sec", {}).items():
            out[f"replay.stage.{stage}_mean_sec"] = stats.get("mean")
    out["replay.peak_rss_mb"] = report.get("peak_rss_mb")
    out["replay.model_load_sec"] = report.get("model_load_sec")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="suite report path")
    parser.add_argument("--label", default="")
    parser.add_argument("--data", default=str(DATA_DIR / "ami"))
    parser.add_argument("--suffix", default="_10min")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument(
        "--stream-sec",
        type=float,
        default=120.0,
        help="replay length per file (0 = full length)",
    )
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument(
        "--reference-config",
        default=str(CONFIGS_DIR / "reference_provider_config.json"),
    )
    parser.add_argument(
        "--dev-vad-config",
        default=str(CONFIGS_DIR / "dev_vad_provider_config.json"),
    )
    parser.add_argument(
        "--with-dev-vad",
        action="store_true",
        help="also run caption latency with the old VAD-on dev config "
        "(secondary, labelled, never gated)",
    )
    parser.add_argument("--caption-audio", default=None)
    parser.add_argument("--caption-seconds", type=float, default=180.0)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--only",
        choices=STEPS,
        nargs="*",
        default=None,
        help="run only these steps (default: all)",
    )
    parser.add_argument(
        "--keep-logs", default=None, help="folder for step logs"
    )
    args = parser.parse_args()

    ensure_hf_token_env()
    steps = set(args.only or STEPS)
    data = Path(args.data)
    caption_audio = Path(
        args.caption_audio or (data / f"ES2004a{args.suffix}.wav")
    )
    if "replay" in steps and not list(data.glob(f"*{args.suffix}.wav")):
        raise SystemExit(
            f"no *{args.suffix}.wav in {data}; run "
            "benchmarks/diarization/prepare_ami_baseline.sh first"
        )
    if "caption" in steps and not caption_audio.exists():
        raise SystemExit(f"caption audio missing: {caption_audio}")

    log_dir = Path(
        args.keep_logs or tempfile.mkdtemp(prefix="diarization_suite_")
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    limits = resource_limits()
    hygiene_before = hygiene_check(limits)
    suite_started = time.perf_counter()
    durations: dict[str, float] = {}

    report: dict = {
        "label": args.label,
        "generated_at": now_iso(),
        "code_revision": git_rev(ROOT),
        "code_dirty": git_dirty(ROOT),
        "environment": environment_info("cpu"),
        "resource_limits": limits,
        "hygiene": {"before": hygiene_before},
        "config": {
            "reference_config": rel_path(args.reference_config),
            "dev_vad_config": (
                rel_path(args.dev_vad_config) if args.with_dev_vad else None
            ),
            "data": rel_path(data),
            "suffix": args.suffix,
            "files": args.files,
            "stream_sec": args.stream_sec,
            "skip_offline": args.skip_offline,
            "caption_audio": rel_path(caption_audio),
            "caption_seconds": args.caption_seconds,
            "chunk_ms": args.chunk_ms,
            "steps": sorted(steps),
        },
        "logs": rel_path(log_dir),
    }

    bench = BENCH_DIR.relative_to(ROOT)

    if "warmup" in steps:
        warm_out = log_dir / "warmup.json"
        durations["warmup"] = run_step(
            "warmup", [str(bench / "warmup.py")], log_dir
        )
        # warmup.py prints one JSON line last
        last = (
            (log_dir / "warmup.log")
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )
        try:
            report["warmup"] = json.loads(last[-1]) if last else {}
        except json.JSONDecodeError:
            report["warmup"] = {}
        write_json(warm_out, report["warmup"])

    if "replay" in steps:
        replay_out = log_dir / "replay.json"
        argv = [
            str(bench / "benchmark_baseline.py"),
            "--data",
            str(data),
            "--suffix",
            args.suffix,
            "--stream-sec",
            str(args.stream_sec),
            "--out",
            str(replay_out),
            "--label",
            args.label,
            "--no-hygiene",
        ]
        if args.files:
            argv += ["--files", *args.files]
        if args.skip_offline:
            argv.append("--skip-offline")
        if args.threads:
            argv += ["--threads", str(args.threads)]
        durations["replay"] = run_step("replay", argv, log_dir)
        report["replay"] = json.loads(replay_out.read_text(encoding="utf-8"))

    if "caption" in steps:
        runs = [
            ("reference", args.reference_config, "on"),
            ("reference", args.reference_config, "off"),
        ]
        if args.with_dev_vad:
            runs += [
                ("secondary_dev_vad", args.dev_vad_config, "on"),
                ("secondary_dev_vad", args.dev_vad_config, "off"),
            ]
        report["caption_latency"] = {}
        for group, config_path, mode in runs:
            name = f"caption_{group}_{mode}"
            out_path = log_dir / f"{name}.json"
            argv = [
                str(bench / "caption_latency.py"),
                "--provider-config",
                str(config_path),
                "--audio",
                str(caption_audio),
                "--diarization",
                mode,
                "--seconds",
                str(args.caption_seconds),
                "--chunk-ms",
                str(args.chunk_ms),
                "--service-log",
                str(log_dir / f"{name}.service.log"),
                "--label",
                f"{args.label} [{group} diarization {mode}]".strip(),
                "--no-hygiene",
                "--out",
                str(out_path),
            ]
            durations[name] = run_step(name, argv, log_dir)
            report["caption_latency"].setdefault(group, {})[mode] = json.loads(
                out_path.read_text(encoding="utf-8")
            )

    report["hygiene"]["after"] = system_state()
    report["durations_sec"] = {k: round(v, 1) for k, v in durations.items()}
    report["suite_wall_sec"] = round(time.perf_counter() - suite_started, 1)

    key_metrics: dict = {}
    key_metrics.update(replay_key_metrics(report.get("replay")))
    reference_runs = (
        _get(report, "caption_latency", "reference", default={}) or {}
    )
    key_metrics.update(
        caption_key_metrics("caption.on", reference_runs.get("on"))
    )
    key_metrics.update(
        caption_key_metrics("caption.off", reference_runs.get("off"))
    )
    report["key_metrics"] = key_metrics
    report["hygiene"]["clean"] = hygiene_before["clean"]
    report["hygiene"]["warnings"] = hygiene_before["warnings"]

    out = Path(args.out)
    write_json(out, report)
    print(f"\nWrote {out}")
    print(f"baseline key: {report['environment']['baseline_key']}")
    if hygiene_before["warnings"]:
        print("HYGIENE WARNINGS (recorded in the report):")
        for line in hygiene_before["warnings"]:
            print(f"  - {line}")
    print("\nkey metrics:")
    for key in sorted(key_metrics):
        print(f"  {key:60s} {key_metrics[key]}")


if __name__ == "__main__":
    main()
