"""
Regression gate: compares a suite report's key metrics against a committed
baseline and fails when any gated metric is worse than the baseline beyond
its tolerance.

A metric regresses when it moves in the bad direction by more than
max(abs, rel * |baseline|) according to baselines/gate_rules.json. Metrics
without a rule are printed for information and never fail the gate.
Metrics missing on either side are reported as SKIPPED.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/compare_baseline.py \
      --report benchmarks/diarization/results/suite.json
  # --baseline auto (default) picks baselines/<environment.baseline_key>.json
  uv run python benchmarks/diarization/compare_baseline.py \
      --report ... --baseline benchmarks/diarization/baselines/linux-cpu-4c8g.json
Exit code 1 on regression, 2 when no baseline exists for the environment.
"""

# pylint: disable=missing-function-docstring

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import BASELINES_DIR  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(name: str, base, current, rule: dict | None) -> tuple[str, str]:
    """Returns (status, detail) for one metric."""
    if base is None or current is None:
        return "SKIPPED", "missing on one side"
    if not isinstance(base, (int, float)) or not isinstance(
        current, (int, float)
    ):
        return "INFO", ""
    delta = current - base
    if rule is None:
        return "INFO", f"{delta:+.4g}"
    tolerance = max(
        float(rule.get("abs", 0.0)), float(rule.get("rel", 0.0)) * abs(base)
    )
    worse = delta if rule.get("direction", "lower") == "lower" else -delta
    if worse > tolerance:
        return "REGRESSION", f"{delta:+.4g} beyond tolerance {tolerance:.4g}"
    if worse < -tolerance:
        return "IMPROVED", f"{delta:+.4g}"
    return "OK", f"{delta:+.4g} within {tolerance:.4g}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--baseline", default="auto")
    parser.add_argument(
        "--rules", default=str(BASELINES_DIR / "gate_rules.json")
    )
    parser.add_argument(
        "--allow-dirty-hygiene",
        action="store_true",
        help="do not warn when the report was taken on a machine with "
        "hygiene warnings (swap in use, high load)",
    )
    args = parser.parse_args()

    report = load(Path(args.report))
    key = report.get("environment", {}).get("baseline_key", "unknown")
    if args.baseline == "auto":
        baseline_path = BASELINES_DIR / f"{key}.json"
    else:
        baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(
            f"no baseline for environment {key!r} at {baseline_path}; "
            "save one with run_suite.py --out <that path> once the numbers "
            "are the ones you want to hold the line at."
        )
        return 2
    baseline = load(baseline_path)
    rules = load(Path(args.rules)).get("rules", {})

    base_key = baseline.get("environment", {}).get("baseline_key")
    if base_key and base_key != key:
        print(
            f"WARNING: report environment {key!r} differs from baseline "
            f"environment {base_key!r}; numbers are not comparable."
        )
    for side, doc in (("baseline", baseline), ("report", report)):
        warnings = doc.get("hygiene", {}).get("warnings") or []
        if warnings and not args.allow_dirty_hygiene:
            print(f"NOTE: {side} was taken with hygiene warnings: {warnings}")

    base_metrics = baseline.get("key_metrics", {})
    cur_metrics = report.get("key_metrics", {})
    names = sorted(set(base_metrics) | set(cur_metrics))
    rows = []
    regressions = 0
    for name in names:
        status, detail = verdict(
            name, base_metrics.get(name), cur_metrics.get(name), rules.get(name)
        )
        if status == "REGRESSION":
            regressions += 1
        rows.append(
            (
                name,
                base_metrics.get(name),
                cur_metrics.get(name),
                status,
                detail,
            )
        )

    print(
        f"baseline {baseline_path.name} @ {baseline.get('code_revision')} "
        f"vs report @ {report.get('code_revision')} ({key})\n"
    )
    print(
        f"{'metric':58s} {'baseline':>12s} {'current':>12s}  status      detail"
    )
    for name, base, cur, status, detail in rows:
        gated = "*" if name in rules else " "
        print(
            f"{gated}{name:57s} {_fmt(base):>12s} {_fmt(cur):>12s}  "
            f"{status:10s}  {detail}"
        )
    print("\n* = gated metric")
    if regressions:
        print(f"\nGATE FAILED: {regressions} gated metric(s) regressed")
        return 1
    print("\nGATE PASSED")
    return 0


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


if __name__ == "__main__":
    sys.exit(main())
