"""
Model or pipeline? Measures how well the embedding model separates the
voices of each meeting, independently of the reconciler, so an
under-counted meeting can be attributed to the recording and model (the
voices are not separable with these embeddings) or to the pipeline (they
are, and the reconciler merged them anyway).

Uses a pass cache written by tune_reconciler.py (one segmentation track per
voice per 10 s window, with its raw embedding). Every track at least
`--min-track-sec` long is given the reference speaker that covers most of
it (purity at least `--purity`); then, per meeting, the same-speaker and
different-speaker track pairs are scored with cosine similarity and with
the PLDA log-likelihood ratio that ships with the model, and the best
achievable error is reported for both: the equal-error threshold, the
error at the production match threshold (cosine) and the fraction of
different-speaker pairs the production threshold would merge.

Usage (from transcription_service/):
    uv run python benchmarks/diarization/embedding_separability.py \\
        --cache benchmarks/diarization/results/phase2c/cache_w10_t5.pkl \\
        --out benchmarks/diarization/results/phase2c/separability.json
"""

# pylint: disable=missing-function-docstring,too-many-locals

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    ensure_hf_token_env,
    now_iso,
    rel_path,
    rttm_turns,
    write_json,
)


def load_plda():
    from pyannote.audio.core.plda import PLDA

    token_var = ensure_hf_token_env()
    import os

    plda = PLDA.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        subfolder="plda",
        token=os.environ.get(token_var),
    )
    if plda is None:
        raise SystemExit("could not load the PLDA of community-1")
    return plda


def plda_llr(
    fea_a: np.ndarray, fea_b: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """
    Pairwise two-covariance PLDA log-likelihood ratio in the PLDA space
    (within-class identity, between-class diag(phi)), rows of a against
    rows of b -> (len(a), len(b))
    """
    tot = 1.0 + phi
    det = tot * tot - phi * phi
    cross = (fea_a * (phi / det)) @ fea_b.T
    quad_a = (fea_a**2 * (phi**2 / (tot * det))).sum(axis=1)
    quad_b = (fea_b**2 * (phi**2 / (tot * det))).sum(axis=1)
    const = -0.5 * np.sum(np.log(1.0 - phi**2 / tot**2))
    return cross - 0.5 * (quad_a[:, None] + quad_b[None, :]) + const


def oracle_tracks(file_cache: dict, turns, min_track_sec: float, purity: float):
    """(embedding, duration, oracle speaker) for every pure, long enough track."""
    tracks = []
    for pass_ in file_cache["passes"]:
        by_label: dict[str, list] = {}
        for start, end, label in pass_["segments"]:
            by_label.setdefault(label, []).append((start, end))
        for label, segments in by_label.items():
            embedding = pass_["embeddings"].get(label)
            if embedding is None:
                continue
            duration = sum(e - s for s, e in segments)
            if duration < min_track_sec:
                continue
            votes: dict[str, float] = {}
            for s, e in segments:
                for ts, te, speaker in turns:
                    overlap = min(e, te) - max(s, ts)
                    if overlap > 0:
                        votes[speaker] = votes.get(speaker, 0.0) + overlap
            if not votes:
                continue
            speaker, covered = max(votes.items(), key=lambda kv: kv[1])
            if covered / duration < purity:
                continue
            tracks.append(
                (np.asarray(embedding, dtype=np.float32), duration, speaker)
            )
    return tracks


def error_at(scores_same, scores_diff, threshold):
    false_split = (
        float(np.mean(scores_same < threshold))
        if len(scores_same)
        else float("nan")
    )
    false_merge = (
        float(np.mean(scores_diff >= threshold))
        if len(scores_diff)
        else float("nan")
    )
    return false_split, false_merge


def equal_error(scores_same, scores_diff):
    if not len(scores_same) or not len(scores_diff):
        return None, None
    candidates = np.unique(np.concatenate([scores_same, scores_diff]))
    best = (None, 1.0)
    for threshold in candidates:
        fs, fm = error_at(scores_same, scores_diff, threshold)
        gap = abs(fs - fm)
        if gap < best[1]:
            best = (float(threshold), gap)
    threshold = best[0]
    fs, fm = error_at(scores_same, scores_diff, threshold)
    return threshold, round((fs + fm) / 2, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--min-track-sec", type=float, default=1.0)
    parser.add_argument("--purity", type=float, default=0.7)
    parser.add_argument("--match-threshold", type=float, default=0.4)
    args = parser.parse_args()

    with open(args.cache, "rb") as handle:
        cache = pickle.load(handle)
    plda = load_plda()
    phi = np.asarray(plda.phi, dtype=np.float64)

    report = {
        "generated_at": now_iso(),
        "cache": rel_path(Path(args.cache)),
        "files": {},
    }
    header = (
        f"{'file':16s} {'spk':>3} {'tracks':>6} "
        f"{'cos EER':>8} {'cos thr':>8} {'merge@0.4':>9} {'split@0.4':>9} "
        f"{'plda EER':>8} {'plda thr':>8}  oracle centroid pairs: min cos / max plda"
    )
    print(header)
    for stem, file_cache in cache["files"].items():
        wav = Path(file_cache["wav"])
        turns = rttm_turns(wav.with_suffix(".rttm"))
        tracks = oracle_tracks(
            file_cache, turns, args.min_track_sec, args.purity
        )
        if len(tracks) < 2:
            continue
        raw = np.stack([t[0] for t in tracks]).astype(np.float64)
        speakers = np.array([t[2] for t in tracks])
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        cos = unit @ unit.T
        fea = plda(raw)
        llr = plda_llr(fea, fea, phi)
        iu = np.triu_indices(len(tracks), k=1)
        same = speakers[iu[0]] == speakers[iu[1]]
        cos_same, cos_diff = cos[iu][same], cos[iu][~same]
        llr_same, llr_diff = llr[iu][same], llr[iu][~same]
        cos_thr, cos_eer = equal_error(cos_same, cos_diff)
        llr_thr, llr_eer = equal_error(llr_same, llr_diff)
        fs, fm = error_at(cos_same, cos_diff, args.match_threshold)

        # Oracle centroids: how far apart the real speakers end up when each
        # is averaged over all its tracks (what the reconciler compares to)
        labels = sorted(set(speakers))
        seconds_per_speaker = {
            s: round(float(sum(t[1] for t in tracks if t[2] == s)), 1)
            for s in labels
        }
        reference_seconds = {}
        for ts, te, speaker in turns:
            reference_seconds[speaker] = round(
                reference_seconds.get(speaker, 0.0) + te - ts, 1
            )
        centroids = np.stack([raw[speakers == s].mean(axis=0) for s in labels])
        cu = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        ccos = cu @ cu.T
        cllr = plda_llr(plda(centroids), plda(centroids), phi)
        pairs = []
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if j > i:
                    pairs.append(
                        {
                            "a": a,
                            "b": b,
                            "cos": round(float(ccos[i, j]), 3),
                            "plda_llr": round(float(cllr[i, j]), 1),
                        }
                    )
        entry = {
            "reference_speakers": len(labels),
            "reference_speakers_in_rttm": len(reference_seconds),
            "reference_seconds": reference_seconds,
            "pure_track_seconds": seconds_per_speaker,
            "tracks": len(tracks),
            "cosine": {
                "eer": cos_eer,
                "eer_threshold": cos_thr,
                "false_split_at_match": round(fs, 4),
                "false_merge_at_match": round(fm, 4),
                "same_speaker_median": (
                    round(float(np.median(cos_same)), 3)
                    if len(cos_same)
                    else None
                ),
                "different_speaker_median": (
                    round(float(np.median(cos_diff)), 3)
                    if len(cos_diff)
                    else None
                ),
            },
            "plda": {
                "eer": llr_eer,
                "eer_threshold": llr_thr,
                "same_speaker_median": (
                    round(float(np.median(llr_same)), 2)
                    if len(llr_same)
                    else None
                ),
                "different_speaker_median": (
                    round(float(np.median(llr_diff)), 2)
                    if len(llr_diff)
                    else None
                ),
            },
            "oracle_centroid_pairs": pairs,
        }
        report["files"][stem] = entry
        min_cos = min((p["cos"] for p in pairs), default=None)
        max_llr = max((p["plda_llr"] for p in pairs), default=None)
        print(
            f"{stem:16s} {len(labels):>3} {len(tracks):>6} "
            f"{cos_eer if cos_eer is not None else float('nan'):8.3f} {cos_thr if cos_thr is not None else float('nan'):8.3f} "
            f"{fm:9.3f} {fs:9.3f} "
            f"{llr_eer if llr_eer is not None else float('nan'):8.3f} {llr_thr if llr_thr is not None else float('nan'):8.1f}  "
            f"{min_cos} / {max_llr}"
        )
        print(
            f"{'':16s} reference seconds {reference_seconds}; pure track seconds {seconds_per_speaker}"
        )
    if args.out:
        write_json(Path(args.out), report)
        print(f"wrote {rel_path(Path(args.out))}")


if __name__ == "__main__":
    main()
