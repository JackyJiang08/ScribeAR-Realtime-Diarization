"""
Phase 2a acceptance check: absolute targets, measured in the Linux CPU
reference container, that compare_baseline.py's relative gate cannot
express.

Reads a suite report (run_suite.py) and the committed Phase 2 baseline for
its environment, evaluates every target in baselines/phase2a_acceptance.json
and prints a table with the margin by which each passed or failed. Exit code
1 when any target is missed. A missed target is reported by how much; it is
never relaxed here.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/acceptance.py \
      --report benchmarks/diarization/results/<date>_suite.json
"""

# pylint: disable=missing-function-docstring,too-many-branches
# pylint: disable=too-many-locals,too-many-statements

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import BASELINES_DIR  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def evaluate(report: dict, baseline: dict | None, targets: dict) -> list:
    """
    One row per target: (name, value, limit, status, note). A target is a
    dict with `metric` (key_metrics name), `max` or `min` (absolute), or
    `max_relative_to` (another metric) with `ratio` and optional `plus`, or
    `max_vs_baseline` with `abs` slack.
    """
    metrics = report.get("key_metrics", {})
    base_metrics = (baseline or {}).get("key_metrics", {})
    rows = []
    for name, target in targets.items():
        value = metrics.get(target["metric"])
        limit = None
        note = target.get("note", "")
        if value is None:
            rows.append((name, None, None, "SKIPPED", "metric missing"))
            continue
        if "max" in target:
            limit = float(target["max"])
            ok = value <= limit
        elif "min" in target:
            limit = float(target["min"])
            ok = value >= limit
        elif "max_relative_to" in target:
            other = metrics.get(target["max_relative_to"])
            if other is None:
                rows.append(
                    (name, value, None, "SKIPPED", "reference metric missing")
                )
                continue
            limit = other * float(target.get("ratio", 1.0)) + float(
                target.get("plus", 0.0)
            )
            ok = value <= limit
        elif "max_vs_baseline" in target:
            base = base_metrics.get(target["max_vs_baseline"])
            if base is None:
                rows.append((name, value, None, "SKIPPED", "no baseline"))
                continue
            limit = base + float(target.get("abs", 0.0))
            ok = value <= limit
        else:
            rows.append((name, value, None, "SKIPPED", "no rule"))
            continue
        if ok:
            margin = abs(limit - value)
            rows.append((name, value, limit, "PASS", f"margin {margin:.3g}"))
        else:
            miss = value - limit
            relative = (miss / limit * 100.0) if limit else float("inf")
            rows.append(
                (
                    name,
                    value,
                    limit,
                    "MISSED",
                    f"by {miss:+.3g} ({relative:+.0f}%). {note}".strip(),
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--baseline", default="auto")
    parser.add_argument(
        "--targets", default=str(BASELINES_DIR / "phase2a_acceptance.json")
    )
    args = parser.parse_args()

    report = load(Path(args.report))
    key = report.get("environment", {}).get("baseline_key", "unknown")
    baseline_path = (
        BASELINES_DIR / f"{key}.json"
        if args.baseline == "auto"
        else Path(args.baseline)
    )
    baseline = load(baseline_path) if baseline_path.exists() else None
    targets = load(Path(args.targets))["targets"]
    rows = evaluate(report, baseline, targets)

    print(
        f"Phase 2a acceptance for report @ {report.get('code_revision')} "
        f"({key}; baseline "
        f"{baseline.get('code_revision') if baseline else 'none'})\n"
    )
    print(f"{'target':52s} {'value':>10s} {'limit':>10s}  status   note")
    missed = 0
    for name, value, limit, status, note in rows:
        if status == "MISSED":
            missed += 1
        print(
            f"{name:52s} {_fmt(value):>10s} {_fmt(limit):>10s}  {status:7s}  {note}"
        )
    if key != "linux-cpu-4c8g":
        print(
            "\nNOTE: the acceptance environment is the Linux CPU reference "
            "container (linux-cpu-4c8g); this report is from another "
            "environment and is informational."
        )
    if missed:
        print(f"\nACCEPTANCE: {missed} target(s) missed")
        return 1
    print("\nACCEPTANCE: all targets met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
