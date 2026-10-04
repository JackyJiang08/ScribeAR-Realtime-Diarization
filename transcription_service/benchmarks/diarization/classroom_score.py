"""
Scores the synthetic classroom case: does every question get a label of
its own, and how often is a question attributed to the instructor?

Reads a replay report written by benchmark_baseline.py with
`--keep-hypotheses` over data/classroom (the settled label timeline, what a
viewer ends up seeing) and the case's question list. For every question the
label covering most of it is compared with the instructor's label (the
label covering most of the lecture time):

- `questions_own_label`: labelled with something other than the
  instructor's label
- `questions_attributed_to_instructor`: the instructor's label won
- `questions_unlabelled`: no label covered the question
- `question_seconds_as_instructor_fraction`: seconds of question speech
  labelled as the instructor, over all question speech
- per questioner: whether their questions share one label, and whether that
  label is theirs alone; the noisy questioner is reported separately

Usage (from transcription_service/):
    uv run python benchmarks/diarization/classroom_score.py \\
        --report benchmarks/diarization/results/<date>_classroom.json
Prints the table and, with --out, writes the scores as JSON; the suite
folds them into key_metrics as classroom.*.
"""

# pylint: disable=missing-function-docstring,too-many-locals

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import DATA_DIR, now_iso, rel_path, write_json  # noqa: E402


def covering(timeline, start: float, end: float) -> dict[str, float]:
    """Seconds of every label inside [start, end)."""
    seconds: dict[str, float] = {}
    for s, e, label in timeline:
        overlap = min(e, end) - max(s, start)
        if overlap > 0:
            seconds[label] = seconds.get(label, 0.0) + overlap
    return seconds


def majority(seconds: dict[str, float]) -> str | None:
    return max(seconds.items(), key=lambda kv: kv[1])[0] if seconds else None


def score(report: dict, questions_doc: dict) -> dict:
    entry = next(
        f for f in report["files"] if f["file"].startswith("classroom")
    )
    timeline = entry["streaming"].get("settled_timeline")
    if timeline is None:
        raise SystemExit(
            "report has no settled_timeline; replay with --keep-hypotheses"
        )
    questions = questions_doc["questions"]
    duration = float(questions_doc["duration_sec"])
    instructor = questions_doc["instructor"]

    # Lecture spans: everything outside the questions
    cursor = 0.0
    lecture_seconds: dict[str, float] = {}
    for q in sorted(questions, key=lambda q: q["start"]):
        for label, sec in covering(timeline, cursor, q["start"]).items():
            lecture_seconds[label] = lecture_seconds.get(label, 0.0) + sec
        cursor = q["end"]
    for label, sec in covering(timeline, cursor, duration).items():
        lecture_seconds[label] = lecture_seconds.get(label, 0.0) + sec
    instructor_label = majority(lecture_seconds)
    lecture_total = sum(lecture_seconds.values())

    rows = []
    own = attributed = unlabelled = 0
    q_seconds_total = q_seconds_as_instructor = 0.0
    per_questioner: dict[str, list] = {}
    for q in questions:
        seconds = covering(timeline, q["start"], q["end"])
        label = majority(seconds)
        length = q["end"] - q["start"]
        as_instructor = seconds.get(instructor_label, 0.0)
        q_seconds_total += length
        q_seconds_as_instructor += as_instructor
        if label is None:
            verdict = "unlabelled"
            unlabelled += 1
        elif label == instructor_label:
            verdict = "instructor"
            attributed += 1
        else:
            verdict = "own"
            own += 1
        rows.append(
            {
                **q,
                "label": label,
                "verdict": verdict,
                "seconds_as_instructor": round(as_instructor, 2),
                "labels_seen": {k: round(v, 2) for k, v in seconds.items()},
            }
        )
        per_questioner.setdefault(q["speaker"], []).append(label)

    questioners = {}
    labels_by_questioner = {}
    for speaker, labels in per_questioner.items():
        used = [l for l in labels if l is not None and l != instructor_label]
        consistent = len(set(used)) == 1 and len(used) == len(labels)
        labels_by_questioner[speaker] = set(used)
        questioners[speaker] = {
            "labels": labels,
            "consistent_own_label": consistent,
            "noisy": speaker == questions_doc["noisy_questioner"],
        }
    for speaker, info in questioners.items():
        others = set().union(
            *(v for k, v in labels_by_questioner.items() if k != speaker)
        )
        info["label_shared_with_other_questioner"] = bool(
            labels_by_questioner[speaker] & others
        )

    noisy = [r for r in rows if r["noisy"]]
    clean = [r for r in rows if not r["noisy"]]
    total = len(rows)
    return {
        "instructor": instructor,
        "instructor_label": instructor_label,
        "instructor_label_share_of_lecture": (
            round(lecture_seconds.get(instructor_label, 0.0) / lecture_total, 3)
            if lecture_total
            else None
        ),
        "lecture_labels": {k: round(v, 1) for k, v in lecture_seconds.items()},
        "questions_total": total,
        "questions_own_label": own,
        "questions_attributed_to_instructor": attributed,
        "questions_unlabelled": unlabelled,
        "questions_own_label_fraction": (
            round(own / total, 3) if total else None
        ),
        "questions_as_instructor_fraction": (
            round(attributed / total, 3) if total else None
        ),
        "question_seconds_as_instructor_fraction": (
            round(q_seconds_as_instructor / q_seconds_total, 3)
            if q_seconds_total
            else None
        ),
        "clean_questions_own_label_fraction": (
            round(sum(r["verdict"] == "own" for r in clean) / len(clean), 3)
            if clean
            else None
        ),
        "noisy_questions_own_label_fraction": (
            round(sum(r["verdict"] == "own" for r in noisy) / len(noisy), 3)
            if noisy
            else None
        ),
        "questioners_with_consistent_own_label": sum(
            1 for q in questioners.values() if q["consistent_own_label"]
        ),
        "questioners_sharing_a_label": sum(
            1
            for q in questioners.values()
            if q["label_shared_with_other_questioner"]
        ),
        "labels_minted": entry["streaming"]["session_labels_minted"],
        "settled_der": entry["streaming"]["settled"]["der"]["value"],
        "settled_confusion": entry["streaming"]["settled"]["der"].get(
            "confusion"
        ),
        "questioners": questioners,
        "questions": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--questions",
        default=str(DATA_DIR / "classroom" / "classroom.questions.json"),
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    result = score(report, questions)
    print(
        f"instructor {result['instructor']} -> {result['instructor_label']} "
        f"({result['instructor_label_share_of_lecture']} of lecture time); "
        f"labels minted {result['labels_minted']}; settled DER {result['settled_der']}"
    )
    print(
        f"{'#':>2} {'speaker':8s} {'noisy':5s} {'start':>7} {'len':>5} {'label':8s} verdict"
    )
    for r in result["questions"]:
        print(
            f"{r['index']:>2} {r['speaker']:8s} {str(r['noisy']):5s} {r['start']:7.1f} "
            f"{r['length_sec']:5.1f} {str(r['label']):8s} {r['verdict']}"
        )
    print(
        f"own label {result['questions_own_label']}/{result['questions_total']} "
        f"({result['questions_own_label_fraction']}), as instructor "
        f"{result['questions_attributed_to_instructor']}, unlabelled "
        f"{result['questions_unlabelled']}; clean {result['clean_questions_own_label_fraction']}, "
        f"noisy {result['noisy_questions_own_label_fraction']}; questioners with one own label "
        f"{result['questioners_with_consistent_own_label']}/3"
    )
    if args.out:
        write_json(
            Path(args.out),
            {
                "generated_at": now_iso(),
                "report": rel_path(Path(args.report)),
                **result,
            },
        )
        print(f"wrote {rel_path(Path(args.out))}")


if __name__ == "__main__":
    main()
