"""
Long-session soak of the real service: streams the concatenated AMI soak
recording (prepare_soak.py) at real time through a running transcription
service with diarization on, for an hour or two, and reports what a long
classroom session would show:

  - memory: RSS of every worker process sampled once a second, with the
    growth from the first steady 10 minutes to the last 10 minutes
  - label drift: per 5 min bin, the session label covering most of each
    reference speaker's finalized words, and the swaps between bins
  - audio dropped by the caption job (`audio_dropped_buffer_full_seconds`,
    the counter a diarization-off run of the same clip is compared against),
    diarization audio skipped, dropped periods, failed passes
  - a mid-run kill of the diarization worker (`--kill-worker-at-sec`): the
    gap in caption messages around it, how long until the first speaker
    label after it, and the replacement worker's model load time from the
    service log
  - model load times from the service log (start-up and after the kill)

Built on caption_latency.py's service runner and message collector. Runs in
the Linux reference container through docker/run_in_docker.sh:
  BENCH_SCRIPT=soak_service.py benchmarks/diarization/docker/run_in_docker.sh \\
      --minutes 120 --kill-worker-at-sec 3600 \\
      --out /app/benchmarks/diarization/results/<name>.json
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-branches
# pylint: disable=import-outside-toplevel,missing-function-docstring
# pylint: disable=too-many-instance-attributes,protected-access

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
    DATA_DIR,
    ROOT,
    SAMPLE_RATE,
    ensure_hf_token_env,
    environment_info,
    git_dirty,
    git_rev,
    hygiene_check,
    load_audio,
    now_iso,
    process_tree_stats,
    rel_path,
    summarize,
    write_json,
)
from caption_latency import (  # noqa: E402
    LatencyCollector,
    ServiceProcess,
    free_port,
    prepare_config,
    stream,
    summarize_counters,
    counter_deltas,
    worker_processes,
)

BENCH_DIR = Path(__file__).resolve().parent
LOAD_LINE = re.compile(r"loaded successfully in ([0-9.]+)s")
WORKER_MIN_RSS_MB = 200.0


class SamplingService(ServiceProcess):
    """
    The service runner with a per-second RSS time series per process and the
    service's own view of the moment a worker was replaced
    """

    def __init__(self, *args, **kwargs):
        self.timeline: list[tuple[float, dict[int, float]]] = []
        self._t0 = time.perf_counter()
        super().__init__(*args, **kwargs)

    def _sample(self):
        while not self._stop.is_set():
            try:
                now = time.perf_counter()
                stats = process_tree_stats(self.process.pid)
                self.rss_samples.append(stats["total_mb"])
                per_pid = {}
                for pid, reading in stats["per_pid"].items():
                    per_pid[pid] = reading["rss_mb"]
                    entry = self.per_pid.setdefault(
                        pid,
                        {
                            "rss_first_mb": reading["rss_mb"],
                            "rss_max_mb": 0.0,
                            "cpu_first": reading["cpu_sec"],
                            "t_first": now,
                            "cpu_last": reading["cpu_sec"],
                            "t_last": now,
                        },
                    )
                    entry["rss_max_mb"] = max(
                        entry["rss_max_mb"], reading["rss_mb"]
                    )
                    entry["cpu_last"] = reading["cpu_sec"]
                    entry["t_last"] = now
                self.timeline.append((now - self._t0, per_pid))
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            self._stop.wait(1.0)

    def worker_pids(self) -> list[int]:
        """Worker pids in spawn order: caption worker first, diarization last."""
        stats = process_tree_stats(self.process.pid)
        pids = [
            pid
            for pid, reading in sorted(stats["per_pid"].items())
            if pid != self.process.pid
            and reading["rss_mb"] >= WORKER_MIN_RSS_MB
        ]
        return pids

    def worker_pids_from_log(self, worker_id: int) -> list[int]:
        """
        The pids that have logged as `worker_id`, oldest first, read from the
        service log (worker log records carry the worker's own pid). The last
        one is the live worker; an earlier one was replaced.
        """
        try:
            lines = self.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return []
        pids: list[int] = []
        needle = f'"worker_id":{worker_id}'
        for line in lines:
            if needle not in line:
                continue
            match = re.search(r'"pid":(\d+)', line)
            if match:
                pid = int(match.group(1))
                if pid not in pids:
                    pids.append(pid)
        return pids


class SoakCollector(LatencyCollector):
    """
    The latency collector plus what the soak needs: every finalized word
    with its stream time, label and arrival, and the arrival of every
    caption message
    """

    def __init__(self, chunk_sec: float):
        super().__init__(chunk_sec)
        # (perf time, cumulative audio the caption job dropped because its
        # buffer was full) sampled from /metrics/status during the run.
        # Whisper's word times count only the audio it kept, so a word's
        # place on the stream (and in the reference RTTM) is its Whisper
        # time plus the audio dropped before it; the session's attacher
        # applies the same shift when it looks labels up. Without this a
        # long run under load scores drift against the wrong speakers.
        self.drop_samples: list[tuple[float, float]] = []
        self.message_arrivals: list[float] = []
        # (start, end, text, speaker_or_None, arrival) per finalized word
        self.final_words_timed: list[tuple] = []
        self.label_arrivals: list[float] = []
        # (word end time in the stream, arrival) for every labelled word seen
        # by any route, so "labels resumed" can be judged on audio recorded
        # after a worker kill rather than on coverage known before it
        self.label_events: list[tuple[float, float]] = []
        self._word_index: dict[tuple, int] = {}

    def clock_shift(self, arrival: float) -> float:
        """Audio the caption job had dropped by `arrival`, in seconds."""
        shift = 0.0
        for at, dropped in self.drop_samples:
            if at > arrival:
                break
            shift = dropped
        return shift

    def on_message(self, payload: dict, arrival: float):
        if payload.get("type") == "transcript":
            self.message_arrivals.append(arrival)
            shift = self.clock_shift(arrival)
            final = payload.get("final")
            if final and final.get("text"):
                texts = final["text"]
                starts = final.get("starts") or [None] * len(texts)
                ends = final.get("ends") or [None] * len(texts)
                speakers = final.get("speakers") or [None] * len(texts)
                for text, start, end, speaker in zip(
                    texts, starts, ends, speakers
                ):
                    if end is None:
                        continue
                    key = (round(float(end), 2), text)
                    if key in self._word_index:
                        continue
                    self._word_index[key] = len(self.final_words_timed)
                    self.final_words_timed.append(
                        [
                            (
                                float(start)
                                if start is not None
                                else float(end) - 0.3
                            )
                            + shift,
                            float(end) + shift,
                            text,
                            speaker,
                            arrival,
                        ]
                    )
                    if speaker is not None:
                        self.label_arrivals.append(arrival)
                        self.label_events.append((float(end) + shift, arrival))
            in_progress = payload.get("in_progress")
            if in_progress and in_progress.get("text"):
                ends = in_progress.get("ends") or []
                speakers = in_progress.get("speakers") or []
                labelled = [
                    float(end)
                    for end, speaker in zip(ends, speakers)
                    if end is not None and speaker is not None
                ]
                if labelled:
                    self.label_arrivals.append(arrival)
                    self.label_events.extend(
                        (end + shift, arrival) for end in labelled
                    )
        elif payload.get("type") == "speakers_update":
            keys = self.sequence_words.get(payload.get("sequence_id")) or []
            for key, speaker in zip(keys, payload.get("speakers") or []):
                if speaker is None:
                    continue
                index = self._word_index.get(key)
                if (
                    index is not None
                    and self.final_words_timed[index][3] is None
                ):
                    self.final_words_timed[index][3] = speaker
                self.label_arrivals.append(arrival)
                self.label_events.append(
                    (float(key[0]) + self.clock_shift(arrival), arrival)
                )
        super().on_message(payload, arrival)


def load_reference(rttm: Path) -> list[tuple[float, float, str]]:
    turns = []
    for line in rttm.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        turns.append((start, start + float(parts[4]), parts[7]))
    return sorted(turns)


def reference_speaker_at(turns, start: float, end: float) -> str | None:
    """Reference speaker overlapping [start, end) the most, if any."""
    best, best_overlap = None, 0.0
    for t_start, t_end, speaker in turns:
        if t_end <= start:
            continue
        if t_start >= end:
            break
        overlap = min(end, t_end) - max(start, t_start)
        if overlap > best_overlap:
            best, best_overlap = speaker, overlap
    return best


def drift_report(words, turns, bin_sec: float, streamed_sec: float) -> dict:
    """
    Per bin: for each reference speaker, the hypothesis label covering most
    of their finalized words; swaps between consecutive bins where the
    speaker has at least 5 labelled words in both.
    """
    bins = int(streamed_sec // bin_sec) + 1
    per_bin: list[dict[str, Counter]] = [
        defaultdict(Counter) for _ in range(bins)
    ]
    labelled = unlabelled = 0
    for start, end, _text, speaker, _arrival in words:
        if end > streamed_sec:
            continue
        reference = reference_speaker_at(turns, start, end)
        if reference is None:
            continue
        index = min(int(((start + end) / 2) // bin_sec), bins - 1)
        if speaker is None:
            unlabelled += 1
            per_bin[index][reference]["<none>"] += 1
            continue
        labelled += 1
        per_bin[index][reference][speaker] += 1
    rows = []
    swaps = 0
    swap_events = []
    previous: dict[str, str] = {}
    for index, counters in enumerate(per_bin):
        majority = {}
        for reference, counter in counters.items():
            labelled_only = Counter(
                {k: v for k, v in counter.items() if k != "<none>"}
            )
            if sum(labelled_only.values()) < 5:
                continue
            label, _ = labelled_only.most_common(1)[0]
            majority[reference] = label
            if reference in previous and previous[reference] != label:
                swaps += 1
                swap_events.append(
                    {
                        "bin": index,
                        "minute": round(index * bin_sec / 60, 1),
                        "reference_speaker": reference,
                        "from": previous[reference],
                        "to": label,
                    }
                )
        previous.update(majority)
        rows.append(
            {
                "bin": index,
                "start_min": round(index * bin_sec / 60, 1),
                "majority_label": majority,
                "labelled_words": sum(
                    v
                    for c in counters.values()
                    for k, v in c.items()
                    if k != "<none>"
                ),
                "unlabelled_words": sum(
                    c.get("<none>", 0) for c in counters.values()
                ),
            }
        )
    hours = max(streamed_sec / 3600.0, 1e-6)
    return {
        "bin_sec": bin_sec,
        "bins": rows,
        "label_swaps": swaps,
        "label_swaps_per_hour": round(swaps / hours, 2),
        "swap_events": swap_events,
        "final_words_labelled": labelled,
        "final_words_unlabelled": unlabelled,
        "final_words_labelled_fraction": (
            round(labelled / (labelled + unlabelled), 4)
            if labelled + unlabelled
            else None
        ),
    }


def memory_report(
    service: SamplingService,
    pids_before: list[int],
    window_sec: float,
    streamed_sec: float,
) -> dict:
    """
    Per process RSS: mean over the first `window_sec` after the first
    steady minute and over the last `window_sec`, growth between them;
    the diarization worker that was killed is reported up to the kill and
    its replacement from its start.
    """
    out = {"per_pid": {}, "window_sec": window_sec}
    timeline = service.timeline
    if not timeline:
        return out
    end_t = timeline[-1][0]
    for pid in sorted({pid for _, per in timeline for pid in per}):
        series = [(t, per[pid]) for t, per in timeline if pid in per]
        if len(series) < 30:
            continue
        first_t = series[0][0]
        warm = [
            v
            for t, v in series
            if first_t + 60 <= t < first_t + 60 + window_sec
        ]
        last = [v for t, v in series if t >= series[-1][0] - window_sec]
        if not warm or not last:
            continue
        first_mean = sum(warm) / len(warm)
        last_mean = sum(last) / len(last)
        out["per_pid"][str(pid)] = {
            "role": (
                "caption_worker"
                if pids_before and pid == pids_before[0]
                else (
                    "diarization_worker"
                    if pids_before and pid == pids_before[-1]
                    else (
                        "parent"
                        if pid == service.process.pid
                        else "worker_replacement_or_other"
                    )
                )
            ),
            "alive_from_sec": round(first_t, 1),
            "alive_to_sec": round(series[-1][0], 1),
            "rss_first_window_mean_mb": round(first_mean, 1),
            "rss_last_window_mean_mb": round(last_mean, 1),
            "rss_max_mb": round(max(v for _, v in series), 1),
            "growth_mb": round(last_mean - first_mean, 1),
            "growth_fraction": (
                round((last_mean - first_mean) / first_mean, 4)
                if first_mean
                else None
            ),
        }
    totals = [sum(per.values()) for _, per in timeline]
    warm_total = [
        sum(per.values()) for t, per in timeline if 60 <= t < 60 + window_sec
    ]
    last_total = [
        sum(per.values()) for t, per in timeline if t >= end_t - window_sec
    ]
    if warm_total and last_total:
        first_mean = sum(warm_total) / len(warm_total)
        last_mean = sum(last_total) / len(last_total)
        out["service_tree"] = {
            "rss_first_window_mean_mb": round(first_mean, 1),
            "rss_last_window_mean_mb": round(last_mean, 1),
            "rss_max_mb": round(max(totals), 1),
            "growth_fraction": round((last_mean - first_mean) / first_mean, 4),
        }
    out["streamed_sec"] = round(streamed_sec, 1)
    return out


def resume_from_log(log_path: Path) -> dict:
    """
    Seconds between the service noticing the dead worker and re-registering
    the diarization job (its labels resume on the next pass), from the log's
    integer timestamps; the client-side metrics add caption latency on top.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    exited = re.search(r'"time":(\d+)[^\n]*exited unexpectedly', text)
    replaced = re.search(r'"time":(\d+)[^\n]*replaced in ([0-9.]+)s', text)
    reregistered = re.search(
        r'"time":(\d+)[^\n]*re-registered after its worker', text
    )
    out: dict = {}
    if exited and replaced:
        out["worker_replaced_after_sec"] = float(replaced.group(2))
    if exited and reregistered:
        out["job_reregistered_after_sec"] = int(reregistered.group(1)) - int(
            exited.group(1)
        )
    return out


def load_times_from_log(log_path: Path) -> list[float]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [float(m.group(1)) for m in LOAD_LINE.finditer(text)]


async def run(
    args, service: SamplingService, samples, collector: SoakCollector, url: str
) -> dict:
    """Streams the clip and, at the configured time, kills the diarization worker."""
    kill_info: dict = {"requested_at_sec": args.kill_worker_at_sec}
    stream_task = asyncio.create_task(
        stream(
            url,
            samples,
            args.chunk_ms / 1000.0,
            args.minutes * 60.0,
            args.linger_sec,
            collector,
            session_uid="diarization-soak",
        )
    )
    if args.kill_worker_at_sec and args.kill_worker_at_sec > 0:
        await asyncio.sleep(args.kill_worker_at_sec)
        pids = service.worker_pids()
        kill_info["worker_pids_at_kill"] = pids
        from_log = service.worker_pids_from_log(1)
        kill_info["diarization_worker_pids_from_log"] = from_log
        # The diarization worker is worker 1 under the reference config; its
        # pid comes from its own log lines, with the RSS ordering as fallback
        target = (
            from_log[-1] if from_log else (pids[-1] if len(pids) >= 2 else None)
        )
        kill_info["killed_pid"] = target
        if target is not None:
            messages_before = len(collector.message_arrivals)
            labels_before = len(collector.label_arrivals)
            killed_at = time.perf_counter()
            os.kill(target, signal.SIGKILL)
            kill_info["killed_at_perf"] = killed_at
            kill_info["messages_before"] = messages_before
            kill_info["labels_before"] = labels_before
            print(
                f"killed diarization worker pid {target} at {args.kill_worker_at_sec}s",
                file=sys.stderr,
                flush=True,
            )
            # Wait for the replacement to show up as a new large pid
            for _ in range(600):
                await asyncio.sleep(0.5)
                now_pids = service.worker_pids()
                if any(p not in pids for p in now_pids):
                    kill_info["replacement_pid"] = [
                        p for p in now_pids if p not in pids
                    ][-1]
                    kill_info["replacement_seen_after_sec"] = round(
                        time.perf_counter() - killed_at, 1
                    )
                    break
    schedule = await stream_task
    return {"schedule": schedule, "kill": kill_info}


def recovery_report(
    collector: SoakCollector, kill_info: dict, service_log: Path
) -> dict | None:
    killed_at = kill_info.get("killed_at_perf")
    if killed_at is None:
        return None
    arrivals = collector.message_arrivals
    before = [a for a in arrivals if a <= killed_at]
    after = [a for a in arrivals if a > killed_at]
    gaps_before = [b - a for a, b in zip(before[:-1], before[1:])][-60:]
    first_after = after[0] - killed_at if after else None
    labels_after = [a for a in collector.label_arrivals if a > killed_at]
    label_resume = labels_after[0] - killed_at if labels_after else None
    # Labels on audio recorded after the kill: the first one can only come
    # from the replacement worker, so this is when diarization really resumed
    kill_stream_sec = kill_info.get("requested_at_sec") or 0.0
    post_kill = [
        arrival - killed_at
        for end, arrival in collector.label_events
        if end >= kill_stream_sec and arrival > killed_at
    ]
    post_kill_resume = min(post_kill) if post_kill else None
    gap_around_kill = (after[0] - before[-1]) if before and after else None
    loads = load_times_from_log(service_log)
    return {
        "captions_first_message_after_kill_sec": (
            round(first_after, 2) if first_after is not None else None
        ),
        "captions_gap_around_kill_sec": (
            round(gap_around_kill, 2) if gap_around_kill is not None else None
        ),
        "captions_typical_gap_before_kill": summarize(gaps_before, 2),
        "labels_resumed_after_sec": (
            round(label_resume, 2) if label_resume is not None else None
        ),
        "labels_for_post_kill_audio_after_sec": (
            round(post_kill_resume, 2) if post_kill_resume is not None else None
        ),
        "label_events_total": len(collector.label_events),
        "label_events_after_kill": sum(
            1 for _, a in collector.label_events if a > killed_at
        ),
        "label_events_on_post_kill_audio": len(post_kill),
        "replacement_seen_after_sec": kill_info.get(
            "replacement_seen_after_sec"
        ),
        "model_load_times_sec": loads,
        "model_load_after_kill_sec": loads[-1] if len(loads) >= 2 else None,
        **resume_from_log(service_log),
        "killed_pid": kill_info.get("killed_pid"),
        "worker_pids_at_kill": kill_info.get("worker_pids_at_kill"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--data", default=str(DATA_DIR / "soak"))
    parser.add_argument(
        "--provider-config",
        default=str(BENCH_DIR / "configs" / "reference_provider_config.json"),
    )
    parser.add_argument("--provider", default="whisper")
    parser.add_argument("--diarization", choices=["on", "off"], default="on")
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--linger-sec", type=float, default=20.0)
    parser.add_argument("--bin-min", type=float, default=5.0)
    parser.add_argument(
        "--kill-worker-at-sec",
        type=float,
        default=0.0,
        help="SIGKILL the diarization worker this many seconds into the stream (0 = never)",
    )
    parser.add_argument("--memory-window-sec", type=float, default=600.0)
    parser.add_argument("--service-log", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-hygiene", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    token_var = (
        ensure_hf_token_env()
        if args.diarization == "on"
        else "HUGGINGFACE_ACCESS_TOKEN"
    )
    stem = f"soak_{args.minutes}min"
    data = Path(args.data)
    wav = data / f"{stem}.wav"
    if not wav.exists():
        # A longer stream can be served by a longer prepared file
        candidates = sorted(data.glob("soak_*min.wav"))
        if not candidates:
            raise SystemExit(
                f"{wav} missing: run prepare_soak.py --minutes {args.minutes}"
            )
        wav = candidates[-1]
        stem = wav.stem
    rttm = data / f"{stem}.rttm"
    turns = load_reference(rttm)
    samples = load_audio(wav)
    if len(samples) < args.minutes * 60 * SAMPLE_RATE:
        raise SystemExit(f"{wav} is shorter than {args.minutes} min")

    config = prepare_config(
        Path(args.provider_config), args.diarization == "on", args.provider
    )
    hygiene = None if args.no_hygiene else hygiene_check()
    scratch = Path(tempfile.mkdtemp(prefix="soak_service_"))
    config_path = scratch / "provider_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = (
        Path(args.service_log)
        if args.service_log
        else Path(args.out).with_suffix(".service.log")
    )
    port = free_port()
    env_extra = {token_var: os.environ.get(token_var, "")}
    if os.environ.get("HF_TOKEN"):
        env_extra["HF_TOKEN"] = os.environ["HF_TOKEN"]

    print(
        f"starting service (diarization {args.diarization}) on port {port}; streaming {args.minutes} min of {wav.name}",
        file=sys.stderr,
        flush=True,
    )
    service = SamplingService(config_path, port, log_path, env_extra)
    try:
        ready_sec = service.wait_ready(300)
        print(
            f"service ready after {ready_sec:.1f}s", file=sys.stderr, flush=True
        )
        pids_before = service.worker_pids()
        metrics_before = service.metrics()
        collector = SoakCollector(args.chunk_ms / 1000.0)
        url = f"ws://127.0.0.1:{port}/transcription_stream/{args.provider}"
        stop_polling = threading.Event()

        def poll_drops():
            while not stop_polling.is_set():
                snapshot = service.metrics()
                if snapshot is not None:
                    dropped = summarize_counters(snapshot).get(
                        "audioDroppedBufferFullSecondsTotal"
                    )
                    if dropped is not None:
                        collector.drop_samples.append(
                            (time.perf_counter(), float(dropped))
                        )
                stop_polling.wait(5.0)

        poller = threading.Thread(target=poll_drops, daemon=True)
        poller.start()
        service.mark("stream_start")
        wall_start = time.perf_counter()
        outcome = asyncio.run(run(args, service, samples, collector, url))
        wall = time.perf_counter() - wall_start
        stop_polling.set()
        poller.join(timeout=10)
        service.mark("stream_end")
        metrics_after = service.metrics()
        exited = service.process.poll()
        workers = worker_processes(service)
    finally:
        service.stop()

    streamed_sec = args.minutes * 60.0
    deltas = counter_deltas(
        summarize_counters(metrics_before), summarize_counters(metrics_after)
    )
    drift = drift_report(
        collector.final_words_timed, turns, args.bin_min * 60.0, streamed_sec
    )
    memory = memory_report(
        service, pids_before, args.memory_window_sec, streamed_sec
    )
    recovery = recovery_report(collector, outcome["kill"], log_path)
    loads = load_times_from_log(log_path)
    latency = collector.report()
    audio = deltas.get("diarizationAudioSecondsTotal") or 0.0
    seconds = deltas.get("diarizationSecondsTotal") or 0.0
    diarization_worker_pid = (
        str(pids_before[-1]) if len(pids_before) >= 2 else None
    )
    report = {
        "label": args.label,
        "generated_at": now_iso(),
        "code_revision": git_rev(ROOT),
        "code_dirty": git_dirty(ROOT),
        "environment": environment_info("cpu"),
        "hygiene": hygiene,
        "config": {
            "audio": rel_path(wav),
            "reference": rel_path(rttm),
            "minutes": args.minutes,
            "diarization": args.diarization,
            "provider_config": rel_path(args.provider_config),
            "kill_worker_at_sec": args.kill_worker_at_sec,
            "bin_min": args.bin_min,
            "memory_window_sec": args.memory_window_sec,
            "effective_provider_config": config["providers"][args.provider][
                "provider_config"
            ],
            "num_workers": config["num_workers"],
        },
        "service": {
            "ready_after_sec": round(ready_sec, 1),
            "exited_during_run": exited,
            "log": rel_path(log_path),
            "workers": workers,
            "model_load_times_sec": loads,
        },
        "stream": outcome["schedule"],
        "wall_sec": round(wall, 1),
        "latency": latency,
        "memory": memory,
        "drift": drift,
        "caption_clock_shift_sec_at_end": (
            collector.drop_samples[-1][1] if collector.drop_samples else 0.0
        ),
        "caption_clock_samples": len(collector.drop_samples),
        "kill": {
            k: v for k, v in outcome["kill"].items() if k != "killed_at_perf"
        },
        "recovery": recovery,
        "service_counters_delta": deltas,
        "key_metrics": {
            "memory.service_tree_growth_fraction": memory.get(
                "service_tree", {}
            ).get("growth_fraction"),
            "memory.diarization_worker_growth_fraction": (
                memory["per_pid"]
                .get(diarization_worker_pid, {})
                .get("growth_fraction")
                if diarization_worker_pid
                else None
            ),
            "memory.caption_worker_growth_fraction": (
                memory["per_pid"]
                .get(str(pids_before[0]), {})
                .get("growth_fraction")
                if pids_before
                else None
            ),
            "drift.label_swaps": drift["label_swaps"],
            "drift.label_swaps_per_hour": drift["label_swaps_per_hour"],
            "drift.final_words_labelled_fraction": drift[
                "final_words_labelled_fraction"
            ],
            "labels_minted": deltas.get("diarizationLabelsMintedTotal"),
            "reference_speakers": len(
                {t[2] for t in turns if t[0] < streamed_sec}
            ),
            "caption.audio_dropped_buffer_full_seconds": deltas.get(
                "audioDroppedBufferFullSecondsTotal"
            ),
            "caption.dropped_periods": deltas.get("asrDroppedPeriodsTotal"),
            "caption.chunk_id_in_progress_p50": latency["chunk_id_method"][
                "in_progress"
            ].get("p50"),
            "caption.chunk_id_in_progress_p95": latency["chunk_id_method"][
                "in_progress"
            ].get("p95"),
            "diarization.uncovered_seconds": deltas.get(
                "diarizationUncoveredSecondsTotal"
            ),
            "diarization.dropped_periods": deltas.get(
                "diarizationDroppedPeriodsTotal"
            ),
            "diarization.failed_passes": deltas.get("diarizationFailedTotal"),
            "diarization.rtf": round(seconds / audio, 3) if audio else None,
            "diarization.label_changes_after_sent": latency[
                "label_changes_after_sent"
            ],
            "recovery.captions_gap_around_kill_sec": (
                recovery["captions_gap_around_kill_sec"] if recovery else None
            ),
            "recovery.labels_resumed_after_sec": (
                recovery["labels_resumed_after_sec"] if recovery else None
            ),
            "recovery.labels_for_post_kill_audio_after_sec": (
                recovery["labels_for_post_kill_audio_after_sec"]
                if recovery
                else None
            ),
            "recovery.job_reregistered_after_sec": (
                recovery.get("job_reregistered_after_sec") if recovery else None
            ),
            "recovery.model_load_after_kill_sec": (
                recovery["model_load_after_kill_sec"] if recovery else None
            ),
            "model_load_startup_sec": loads[0] if loads else None,
        },
    }
    write_json(Path(args.out), report)
    print(json.dumps(report["key_metrics"], indent=2))
    print(f"\nreport: {args.out}\nservice log: {log_path}")


if __name__ == "__main__":
    main()
