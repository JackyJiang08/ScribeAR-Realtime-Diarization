"""
End-to-end caption latency probe: runs the real transcription service with
a given provider config, streams a recording at it in real time as SAFP
frames (the same instrument shape as upstream's tools/asr-load and the
production kiosk) and measures how long captions take to appear, and - when
diarization is on - how long their speaker labels take to follow.

Two caption-latency methods are reported, both from the audit
(docs/diarization_production_audit.md, section 0):

  chunk-id (PRIMARY) - delay from the send time of the newest chunk listed in
      `in_progress_chunk_ids` / `final_chunk_ids` to the arrival of that
      message. This is exactly what node-server aggregates into its latency
      percentiles (latency-window.ts), so these numbers are comparable with
      the fleet dashboard and the client's latency badge.
  word (secondary) - for each (end_timestamp, text) pair, delay from the send
      time of the chunk containing `end` to the first message showing the
      pair, in `in_progress` or `final`; and separately the first `final`
      showing it. Harsher: every re-timestamped word counts as new.

Speaker labels (Phase 2a) are measured per finalized word: the delay from
the first message that showed the word's text to the first message that gave
it a non-null label, whether that was an in-progress sequence, the final
sequence or a later `speakers_update`. Corrections are counted two ways: a
label that changed between first showing and finalization, and a
`speakers_update` that disagreed with a label already sent (which must be
zero by design).

It also records the service's own counters over the run (dropped periods,
audio dropped because the buffer was full, buffer overflows, the diarization
counters, execution-time and lag histograms) via /metrics/status, and the
CPU and RSS of every process in the service tree, so the diarization
worker's cost can be read off separately from the caption worker's.

`--sessions N` streams the same recording over N concurrent sessions, for
the capacity question.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/caption_latency.py \
      --provider-config benchmarks/diarization/configs/reference_provider_config.json \
      --audio benchmarks/diarization/data/ami/ES2004a_10min.wav \
      --diarization on --seconds 180 --out results/caption_on.json
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-instance-attributes
# pylint: disable=missing-function-docstring,broad-exception-caught
# pylint: disable=too-many-branches,import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-positional-arguments

import argparse
import asyncio
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import (  # noqa: E402
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

sys.path.insert(0, str(ROOT))
from src.shared.utils.audio_frame_protocol import (  # noqa: E402
    encode_audio_frame,
)

API_KEY = "benchmark-api-key"
METRICS_API_KEY = "benchmark-metrics-key"
READINESS_TIMEOUT_SEC = 1200.0

COUNTERS_OF_INTEREST = (
    "jobsCompletedTotal",
    "jobsFailedTotal",
    "asrAudioSecondsTotal",
    "asrDroppedPeriodsTotal",
    "audioDroppedBufferFullTotal",
    "audioDroppedBufferFullSecondsTotal",
    "bufferOverflowTotal",
    "bufferOverflowSecondsTotal",
    "vadNoSpeechTotal",
    "noWordsTotal",
    "temperatureFallbackTotal",
    "repeatedSegmentDetectedTotal",
    "diarizationRunsTotal",
    "diarizationSecondsTotal",
    "diarizationFailedTotal",
    "reconcilerSecondsTotal",
    "diarizationLabelsMintedTotal",
    "diarizationAudioSecondsTotal",
    "diarizationUncoveredSecondsTotal",
    "diarizationDroppedPeriodsTotal",
)
HISTOGRAMS_OF_INTEREST = (
    "asrExecutionMs",
    "asrSchedulingDelayMs",
    "asrRtf",
    "diarizationExecutionMs",
    "diarizationLagMs",
    "diarizationRtf",
)

# A worker process holds at least a Whisper model; anything smaller in the
# service tree is the multiprocess resource tracker or the FastAPI parent.
WORKER_MIN_RSS_MB = 200.0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _parse_override(spec: str) -> tuple[str, object]:
    """`key=value` with a JSON value (falls back to the raw string)."""
    key, _, raw = spec.partition("=")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key, value


def prepare_config(
    path: Path,
    diarization_on: bool,
    provider: str,
    provider_overrides: list[str] | None = None,
    context_overrides: list[str] | None = None,
) -> dict:
    """
    Loads the reference provider config and derives the diarization-off
    variant: `diarization_detector` false, the pyannote context removed and
    the worker count reduced to the workers that still own a context, so the
    off run is exactly upstream's shipped configuration (no pyannote loaded,
    no HuggingFace token needed, one worker).

    `provider_overrides` (`key=json`) patch the provider config and
    `context_overrides` (`context_uid.key=json`) patch a context config, so a
    sweep can vary the diarization window or the pyannote thread count in
    the live service without a config file per point.
    """
    config = json.loads(path.read_text(encoding="utf-8"))
    config.pop("_comment", None)
    if provider not in config["providers"]:
        raise SystemExit(f"provider {provider!r} not in {path}")
    provider_config = config["providers"][provider]["provider_config"]
    if diarization_on:
        provider_config["diarization_detector"] = True
    else:
        provider_config["diarization_detector"] = False
        for key in list(provider_config):
            if key.startswith("diarization_") and key != "diarization_detector":
                provider_config.pop(key)
        config["contexts"] = [
            ctx
            for ctx in config["contexts"]
            if ctx["context_uid"] != "pyannote-diarization"
        ]
        used_workers = sorted(
            {w for ctx in config["contexts"] for w in ctx["worker_ids"]}
        )
        remap = {old: new for new, old in enumerate(used_workers)}
        for ctx in config["contexts"]:
            ctx["worker_ids"] = [remap[w] for w in ctx["worker_ids"]]
        config["num_workers"] = max(1, len(used_workers))
    for spec in provider_overrides or []:
        key, value = _parse_override(spec)
        provider_config[key] = value
    for spec in context_overrides or []:
        key, value = _parse_override(spec)
        uid, _, field = key.partition(".")
        for ctx in config["contexts"]:
            if ctx["context_uid"] == uid:
                ctx["context_config"][field] = value
    return config


def encode_wav_chunk(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class ServiceProcess:
    """
    The transcription service as a child process, with a sampler that
    records RSS and CPU time for every process in its tree once a second.
    """

    def __init__(self, config_path: Path, port: int, log_path: Path, env_extra):
        self.port = port
        self.log_path = log_path
        env = dict(os.environ)
        env.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(port),
                "API_KEY": API_KEY,
                "METRICS_API_KEY": METRICS_API_KEY,
                "WS_INIT_TIMEOUT_SEC": "30",
                "PROVIDER_CONFIG_PATH": str(config_path),
                "LOG_LEVEL": env_extra.get("LOG_LEVEL", "info"),
            }
        )
        env.update(env_extra)
        self._log = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
        self.process = subprocess.Popen(
            [sys.executable, "src/index.py"],
            cwd=str(ROOT),
            env=env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.rss_samples: list[float] = []
        # pid -> {"rss_max_mb", "cpu_first", "cpu_last", "t_first", "t_last",
        #         "rss_first_mb"}
        self.per_pid: dict[int, dict] = {}
        self._mark: dict[str, dict[int, tuple[float, float]]] = {}
        self._stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()

    def _sample(self):
        while not self._stop.is_set():
            try:
                now = time.perf_counter()
                stats = process_tree_stats(self.process.pid)
                self.rss_samples.append(stats["total_mb"])
                for pid, reading in stats["per_pid"].items():
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
            except Exception:
                pass
            self._stop.wait(1.0)

    def mark(self, name: str) -> None:
        """Snapshots every process's CPU seconds under a name."""
        now = time.perf_counter()
        try:
            stats = process_tree_stats(self.process.pid)
        except Exception:
            stats = {"per_pid": {}}
        self._mark[name] = {
            pid: (reading["cpu_sec"], now)
            for pid, reading in stats["per_pid"].items()
        }

    def cpu_between(self, start: str, end: str) -> dict[int, dict]:
        """CPU seconds and cores per process between two marks."""
        out: dict[int, dict] = {}
        first = self._mark.get(start, {})
        last = self._mark.get(end, {})
        for pid, (cpu_end, t_end) in last.items():
            if pid not in first:
                continue
            cpu_start, t_start = first[pid]
            wall = max(1e-6, t_end - t_start)
            out[pid] = {
                "cpu_sec": round(cpu_end - cpu_start, 2),
                "wall_sec": round(wall, 1),
                "cores": round((cpu_end - cpu_start) / wall, 3),
            }
        return out

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout_sec: float) -> float:
        started = time.perf_counter()
        deadline = started + timeout_sec
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise SystemExit(
                    f"service exited with {self.process.returncode} before "
                    f"ready; see {self.log_path}\n{self.log_tail()}"
                )
            try:
                response = httpx.get(
                    f"{self.base_url}/probes/readiness", timeout=2.0
                )
                if response.status_code == 200:
                    return time.perf_counter() - started
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise SystemExit(
            f"service not ready after {timeout_sec}s; {self.log_tail()}"
        )

    def metrics(self) -> dict | None:
        try:
            response = httpx.get(
                f"{self.base_url}/metrics/status",
                headers={"Authorization": f"Bearer {METRICS_API_KEY}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                return response.json()
        except httpx.HTTPError:
            pass
        return None

    def log_tail(self, lines: int = 20) -> str:
        try:
            return "\n".join(
                self.log_path.read_text(encoding="utf-8").splitlines()[-lines:]
            )
        except OSError:
            return ""

    def stop(self):
        self._stop.set()
        self._sampler.join(timeout=3)
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        self._log.close()


def summarize_counters(snapshot: dict | None) -> dict:
    """Sums every series of each counter of interest."""
    out: dict = {}
    if not snapshot:
        return out
    counters = snapshot.get("counters", {})
    for name in COUNTERS_OF_INTEREST:
        series = counters.get(name) or []
        out[name] = round(sum(float(s.get("value", 0)) for s in series), 3)
    return out


def summarize_histograms(snapshot: dict | None) -> dict:
    out: dict = {}
    if not snapshot:
        return out
    histograms = snapshot.get("histograms", {})
    for name in HISTOGRAMS_OF_INTEREST:
        series = histograms.get(name) or []
        if not series:
            continue
        # One provider in the benchmark config, so take the busiest series
        best = max(series, key=lambda s: s.get("count", 0))
        out[name] = {
            k: best.get(k)
            for k in (
                "count",
                "sampleCount",
                "mean",
                "p50",
                "p95",
                "p99",
                "max",
            )
        }
    return out


def counter_deltas(before: dict, after: dict) -> dict:
    return {
        name: round(after.get(name, 0.0) - before.get(name, 0.0), 3)
        for name in COUNTERS_OF_INTEREST
    }


class LatencyCollector:
    """
    Correlates transcript and speakers_update messages with the send times
    of audio chunks, for one session.
    """

    def __init__(self, chunk_sec: float):
        self.chunk_sec = chunk_sec
        self.send_time: dict[int, float] = {}
        self.chunk_id_in_progress: list[float] = []
        self.chunk_id_final: list[float] = []
        self.word_first_shown: dict[tuple, float] = {}
        self.word_finalized: dict[tuple, float] = {}
        self.word_first_speaker: dict[tuple, str | None] = {}
        self.word_final_speaker: dict[tuple, str | None] = {}
        # Wall time the word first carried a non-null label, by any route.
        self.word_label_known: dict[tuple, float] = {}
        self.sequence_words: dict[str, list[tuple]] = {}
        self.sequence_settled: set[str] = set()
        self.label_changes_before_final = 0
        self.label_changes_after_sent = 0
        self.speakers_updates = 0
        self.speakers_updates_unknown_sequence = 0
        self.messages = 0
        self.messages_with_text = 0
        self.final_words = 0
        self.labels_seen: set[str] = set()
        self.first_message_at: float | None = None
        self.last_sent_index = -1

    @staticmethod
    def _chunk_index(chunk_id: str) -> int | None:
        try:
            return int(chunk_id.rsplit("c", 1)[1])
        except (IndexError, ValueError):
            return None

    def on_sent(self, index: int, at: float):
        self.send_time[index] = at
        self.last_sent_index = index

    def _note_label(self, word_key: tuple, speaker, arrival: float) -> None:
        if speaker is None:
            return
        self.labels_seen.add(speaker)
        if word_key not in self.word_label_known:
            self.word_label_known[word_key] = arrival

    def on_message(self, payload: dict, arrival: float):
        kind = payload.get("type")
        if kind == "speakers_update":
            self._on_speakers_update(payload, arrival)
            return
        if kind != "transcript":
            return
        self.messages += 1
        if self.first_message_at is None:
            self.first_message_at = arrival
        for sequence_kind, key in (
            ("in_progress", "in_progress_chunk_ids"),
            ("final", "final_chunk_ids"),
        ):
            ids = payload.get(key) or []
            indices = [i for i in map(self._chunk_index, ids) if i is not None]
            if indices:
                newest = max(indices)
                if newest in self.send_time:
                    latency = arrival - self.send_time[newest]
                    (
                        self.chunk_id_in_progress
                        if sequence_kind == "in_progress"
                        else self.chunk_id_final
                    ).append(latency)
        had_text = False
        for sequence_kind in ("in_progress", "final"):
            sequence = payload.get(sequence_kind)
            if not sequence or not sequence.get("text"):
                continue
            had_text = True
            texts = sequence["text"]
            ends = sequence.get("ends") or [None] * len(texts)
            speakers = sequence.get("speakers") or [None] * len(texts)
            word_keys: list[tuple] = []
            for text, end, speaker in zip(texts, ends, speakers):
                if end is None:
                    word_keys.append((None, text))
                    continue
                word_key = (round(float(end), 2), text)
                word_keys.append(word_key)
                chunk_index = min(
                    int(float(end) // self.chunk_sec), self.last_sent_index
                )
                if chunk_index not in self.send_time:
                    continue
                latency = arrival - self.send_time[chunk_index]
                if word_key not in self.word_first_shown:
                    self.word_first_shown[word_key] = latency
                    self.word_first_speaker[word_key] = speaker
                self._note_label(word_key, speaker, arrival)
                if sequence_kind == "final":
                    if word_key not in self.word_finalized:
                        self.word_finalized[word_key] = latency
                        self.word_final_speaker[word_key] = speaker
                        self.final_words += 1
                        first = self.word_first_speaker.get(word_key)
                        if (
                            first is not None
                            and speaker is not None
                            and first != speaker
                        ):
                            self.label_changes_before_final += 1
            if sequence_kind == "final" and sequence.get("sequence_id"):
                self.sequence_words[sequence["sequence_id"]] = word_keys
        if had_text:
            self.messages_with_text += 1

    def _on_speakers_update(self, payload: dict, arrival: float) -> None:
        self.speakers_updates += 1
        sequence_id = payload.get("sequence_id")
        keys = self.sequence_words.get(sequence_id)
        if keys is None:
            self.speakers_updates_unknown_sequence += 1
            return
        for word_key, speaker in zip(keys, payload.get("speakers") or []):
            if speaker is None:
                continue
            previous = self.word_final_speaker.get(word_key)
            if previous is not None and previous != speaker:
                self.label_changes_after_sent += 1
                continue
            if previous is None:
                self.word_final_speaker[word_key] = speaker
            self._note_label(word_key, speaker, arrival)
        if payload.get("settled"):
            self.sequence_settled.add(sequence_id)

    def report(self) -> dict:
        label_after_text: list[float] = []
        label_after_final: list[float] = []
        labelled = 0
        for word_key in self.word_finalized:
            known = self.word_label_known.get(word_key)
            if known is None:
                continue
            labelled += 1
            # Both are wall-clock delays from the message that first showed
            # the text (or finalized it) to the message that labelled it; a
            # label that arrived with or before the text is 0.
            shown_at = self.send_time.get(
                min(int(word_key[0] // self.chunk_sec), self.last_sent_index),
                None,
            )
            if shown_at is None:
                continue
            shown_wall = shown_at + self.word_first_shown[word_key]
            final_wall = shown_at + self.word_finalized[word_key]
            label_after_text.append(max(0.0, known - shown_wall))
            label_after_final.append(max(0.0, known - final_wall))
        return {
            "chunk_id_method": {
                "in_progress": summarize(self.chunk_id_in_progress, 3),
                "final": summarize(self.chunk_id_final, 3),
            },
            "word_method": {
                "first_shown": summarize(
                    list(self.word_first_shown.values()), 3
                ),
                "finalized": summarize(list(self.word_finalized.values()), 3),
            },
            "label_latency": {
                "after_text_shown": summarize(label_after_text, 3),
                "after_finalized": summarize(label_after_final, 3),
            },
            "transcript_messages": self.messages,
            "transcript_messages_with_text": self.messages_with_text,
            "speakers_update_messages": self.speakers_updates,
            "speakers_updates_for_unknown_sequence": (
                self.speakers_updates_unknown_sequence
            ),
            "final_sequences_with_id": len(self.sequence_words),
            "final_sequences_settled": len(self.sequence_settled),
            "final_words": self.final_words,
            "final_words_labelled": labelled,
            "final_words_labelled_fraction": (
                round(labelled / self.final_words, 3)
                if self.final_words
                else None
            ),
            "speaker_labels_seen": sorted(self.labels_seen),
            "label_changes_before_final": self.label_changes_before_final,
            "label_changes_before_final_fraction": (
                round(self.label_changes_before_final / self.final_words, 4)
                if self.final_words
                else None
            ),
            "label_changes_after_sent": self.label_changes_after_sent,
        }


async def stream(
    url: str,
    samples: np.ndarray,
    chunk_sec: float,
    seconds: float,
    linger_sec: float,
    collector: LatencyCollector,
    session_uid: str = "diarization-benchmark",
    quiet: bool = False,
) -> dict:
    from websockets.asyncio.client import connect

    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    total_chunks = min(
        int(seconds / chunk_sec), int(np.ceil(len(samples) / chunk_samples))
    )
    schedule: dict = {"late_sends": 0, "max_send_lateness_sec": 0.0}

    async with connect(url, max_size=None) as websocket:
        await websocket.send(json.dumps({"type": "auth", "api_key": API_KEY}))
        await websocket.send(
            json.dumps(
                {
                    "type": "config",
                    "config": {},
                    "session_uid": session_uid,
                    "room_uid": "diarization-benchmark",
                }
            )
        )

        async def receiver():
            try:
                async for message in websocket:
                    arrival = time.perf_counter()
                    if isinstance(message, bytes):
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    collector.on_message(payload, arrival)
            except Exception:
                return

        receiver_task = asyncio.create_task(receiver())
        t0 = time.perf_counter()
        for index in range(total_chunks):
            due = t0 + index * chunk_sec
            now = time.perf_counter()
            if due > now:
                await asyncio.sleep(due - now)
            else:
                lateness = now - due
                if lateness > 0.05:
                    schedule["late_sends"] += 1
                schedule["max_send_lateness_sec"] = max(
                    schedule["max_send_lateness_sec"], lateness
                )
            chunk = samples[index * chunk_samples : (index + 1) * chunk_samples]
            frame = encode_audio_frame(
                f"c{index}",
                encode_wav_chunk(chunk),
                sent_at=time.time() * 1000.0,
            )
            sent_at = time.perf_counter()
            await websocket.send(frame)
            collector.on_sent(index, sent_at)
            if index % 20 == 0 and not quiet:
                print(
                    f"  sent {(index + 1) * chunk_sec:6.1f}s  messages "
                    f"{collector.messages}",
                    file=sys.stderr,
                    flush=True,
                )
        schedule["chunks_sent"] = total_chunks
        schedule["streamed_sec"] = round(total_chunks * chunk_sec, 1)
        await asyncio.sleep(linger_sec)
        receiver_task.cancel()
        try:
            await receiver_task
        except (asyncio.CancelledError, Exception):
            pass
    schedule["max_send_lateness_sec"] = round(
        schedule["max_send_lateness_sec"], 3
    )
    return schedule


async def stream_sessions(
    url: str,
    samples: np.ndarray,
    chunk_sec: float,
    seconds: float,
    linger_sec: float,
    collectors: list[LatencyCollector],
) -> list[dict]:
    """Streams the recording over every collector's session at once."""
    return list(
        await asyncio.gather(
            *(
                stream(
                    url,
                    samples,
                    chunk_sec,
                    seconds,
                    linger_sec,
                    collector,
                    session_uid=f"diarization-benchmark-{index}",
                    quiet=index > 0,
                )
                for index, collector in enumerate(collectors)
            )
        )
    )


def _pool(collectors: list[LatencyCollector], path: tuple[str, ...]) -> dict:
    """Pools one summarized latency list across sessions."""
    values: list[float] = []
    for collector in collectors:
        attr = collector
        for key in path:
            attr = getattr(attr, key)
        values.extend(attr)
    return summarize(values, 3)


def worker_processes(service: ServiceProcess) -> list[dict]:
    """
    The worker processes of the service tree in spawn order (pid order),
    with their CPU and RSS over the streaming window. The FastAPI parent and
    the multiprocess resource tracker are left out by RSS. With the
    reference config the first worker runs captions and the second runs
    diarization.
    """
    cpu = service.cpu_between("stream_start", "stream_end")
    workers = []
    parent = service.process.pid
    for pid in sorted(service.per_pid):
        entry = service.per_pid[pid]
        if pid == parent or entry["rss_max_mb"] < WORKER_MIN_RSS_MB:
            continue
        workers.append(
            {
                "pid": pid,
                "rss_max_mb": round(entry["rss_max_mb"], 1),
                **cpu.get(
                    pid, {"cpu_sec": None, "wall_sec": None, "cores": None}
                ),
            }
        )
    for index, worker in enumerate(workers):
        worker["worker"] = index
    return workers


def run_probe(args) -> dict:
    token_var = ensure_hf_token_env()
    diarization_on = args.diarization == "on"
    config = prepare_config(
        Path(args.provider_config),
        diarization_on,
        args.provider,
        args.provider_set,
        args.context_set,
    )
    hygiene = None if args.no_hygiene else hygiene_check()

    scratch = Path(tempfile.mkdtemp(prefix="caption_latency_"))
    config_path = scratch / "provider_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_path = (
        Path(args.service_log) if args.service_log else scratch / "service.log"
    )

    samples = load_audio(Path(args.audio))
    chunk_sec = args.chunk_ms / 1000.0
    port = args.port or free_port()
    env_extra = {token_var: os.environ.get(token_var, "")}
    if os.environ.get("HF_TOKEN"):
        env_extra["HF_TOKEN"] = os.environ["HF_TOKEN"]

    print(
        f"starting service (diarization {args.diarization}, "
        f"{args.sessions} session(s)) on port {port}",
        file=sys.stderr,
        flush=True,
    )
    service = ServiceProcess(config_path, port, log_path, env_extra)
    try:
        ready_sec = service.wait_ready(READINESS_TIMEOUT_SEC)
        print(
            f"service ready after {ready_sec:.1f}s", file=sys.stderr, flush=True
        )
        rss_ready = process_tree_stats(service.process.pid)
        metrics_before = service.metrics()
        collectors = [LatencyCollector(chunk_sec) for _ in range(args.sessions)]
        url = f"ws://127.0.0.1:{port}/transcription_stream/{args.provider}"
        service.mark("stream_start")
        wall_start = time.perf_counter()
        schedules = asyncio.run(
            stream_sessions(
                url,
                samples,
                chunk_sec,
                args.seconds,
                args.linger_sec,
                collectors,
            )
        )
        wall = time.perf_counter() - wall_start
        service.mark("stream_end")
        metrics_after = service.metrics()
        rss_end = process_tree_stats(service.process.pid)
        exited = service.process.poll()
        workers = worker_processes(service)
    finally:
        service.stop()

    before = summarize_counters(metrics_before)
    after = summarize_counters(metrics_after)
    reports = [collector.report() for collector in collectors]
    latency = reports[0] if args.sessions == 1 else _pooled_report(collectors)
    deltas = counter_deltas(before, after)
    diarization_cost = _diarization_cost(deltas, wall, workers, diarization_on)
    report = {
        "label": args.label,
        "generated_at": now_iso(),
        "code_revision": git_rev(ROOT),
        "code_dirty": git_dirty(ROOT),
        "environment": environment_info("cpu"),
        "hygiene": hygiene,
        "config": {
            "provider_config": rel_path(args.provider_config),
            "provider": args.provider,
            "diarization": args.diarization,
            "sessions": args.sessions,
            "provider_overrides": args.provider_set,
            "context_overrides": args.context_set,
            "effective_provider_config": config["providers"][args.provider][
                "provider_config"
            ],
            "num_workers": config["num_workers"],
            "contexts": [
                {
                    "uid": ctx["context_uid"],
                    "worker_ids": ctx["worker_ids"],
                    "config": ctx["context_config"],
                }
                for ctx in config["contexts"]
            ],
            "audio": rel_path(args.audio),
            "chunk_ms": args.chunk_ms,
            "seconds": args.seconds,
            "linger_sec": args.linger_sec,
        },
        "service": {
            "ready_after_sec": round(ready_sec, 1),
            "exited_during_run": exited,
            "log": rel_path(log_path),
            "rss_mb_after_ready": rss_ready["total_mb"],
            "rss_mb_end": rss_end["total_mb"],
            "rss_mb_peak": round(max(service.rss_samples or [0.0]), 1),
            "rss_mb_growth_during_stream": round(
                rss_end["total_mb"] - rss_ready["total_mb"], 1
            ),
            "rss_samples": len(service.rss_samples),
            "workers": workers,
        },
        "diarization_cost": diarization_cost,
        "stream": {
            "sessions": args.sessions,
            "wall_sec": round(wall, 1),
            "per_session": schedules,
        },
        "latency": latency,
        "per_session_latency": reports if args.sessions > 1 else None,
        "service_counters_delta": deltas,
        "service_histograms_end": summarize_histograms(metrics_after),
    }
    return report


def _pooled_report(collectors: list[LatencyCollector]) -> dict:
    """One latency report pooling every session's samples."""
    reports = [c.report() for c in collectors]
    pooled = json.loads(json.dumps(reports[0]))
    pooled["chunk_id_method"] = {
        "in_progress": _pool(collectors, ("chunk_id_in_progress",)),
        "final": _pool(collectors, ("chunk_id_final",)),
    }
    pooled["word_method"] = {
        "first_shown": summarize(
            [v for c in collectors for v in c.word_first_shown.values()], 3
        ),
        "finalized": summarize(
            [v for c in collectors for v in c.word_finalized.values()], 3
        ),
    }
    for key in (
        "transcript_messages",
        "transcript_messages_with_text",
        "speakers_update_messages",
        "speakers_updates_for_unknown_sequence",
        "final_sequences_with_id",
        "final_sequences_settled",
        "final_words",
        "final_words_labelled",
        "label_changes_before_final",
        "label_changes_after_sent",
    ):
        pooled[key] = sum(r[key] for r in reports)
    pooled["final_words_labelled_fraction"] = (
        round(pooled["final_words_labelled"] / pooled["final_words"], 3)
        if pooled["final_words"]
        else None
    )
    pooled["label_changes_before_final_fraction"] = (
        round(pooled["label_changes_before_final"] / pooled["final_words"], 4)
        if pooled["final_words"]
        else None
    )
    # Pool the per-word label delays by re-deriving them from each report's
    # summary is impossible; recompute from the collectors instead.
    after_text: list[float] = []
    after_final: list[float] = []
    for collector in collectors:
        single = collector.report()["label_latency"]
        # summarize() loses the samples, so rebuild them the same way
        del single
    for collector in collectors:
        for word_key, known in collector.word_label_known.items():
            if word_key not in collector.word_finalized:
                continue
            shown_at = collector.send_time.get(
                min(
                    int(word_key[0] // collector.chunk_sec),
                    collector.last_sent_index,
                )
            )
            if shown_at is None:
                continue
            after_text.append(
                max(
                    0.0,
                    known - (shown_at + collector.word_first_shown[word_key]),
                )
            )
            after_final.append(
                max(
                    0.0, known - (shown_at + collector.word_finalized[word_key])
                )
            )
    pooled["label_latency"] = {
        "after_text_shown": summarize(after_text, 3),
        "after_finalized": summarize(after_final, 3),
    }
    pooled["speaker_labels_seen"] = sorted(
        set().union(*(c.labels_seen for c in collectors))
    )
    return pooled


def _diarization_cost(
    deltas: dict, wall: float, workers: list[dict], diarization_on: bool
) -> dict | None:
    """
    The per-run cost of diarization: compute seconds per second of audio
    received (RTF, from the service's own counters), the diarization
    worker's CPU cores and RSS, and the extra RSS the whole service tree
    carries. The diarization worker is the last worker in spawn order under
    the reference config (pyannote on worker 1).
    """
    if not diarization_on:
        return None
    audio = deltas.get("diarizationAudioSecondsTotal") or 0.0
    seconds = deltas.get("diarizationSecondsTotal") or 0.0
    runs = deltas.get("diarizationRunsTotal") or 0.0
    worker = workers[-1] if len(workers) >= 2 else None
    return {
        "rtf_from_counters": round(seconds / audio, 3) if audio else None,
        "pass_cost_mean_sec": round(seconds / runs, 3) if runs else None,
        "runs": runs,
        "audio_seconds": audio,
        "uncovered_seconds": deltas.get("diarizationUncoveredSecondsTotal"),
        "dropped_periods": deltas.get("diarizationDroppedPeriodsTotal"),
        "failed_passes": deltas.get("diarizationFailedTotal"),
        "labels_minted": deltas.get("diarizationLabelsMintedTotal"),
        "worker_cores_mean": worker["cores"] if worker else None,
        "worker_cpu_sec": worker["cpu_sec"] if worker else None,
        "worker_rss_max_mb": worker["rss_max_mb"] if worker else None,
        "wall_sec": round(wall, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--provider", default="whisper")
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV")
    parser.add_argument("--diarization", choices=["on", "off"], default="on")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--linger-sec", type=float, default=20.0)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument(
        "--provider-set",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="override a provider_config field, e.g. diarization_window_sec=15",
    )
    parser.add_argument(
        "--context-set",
        action="append",
        default=[],
        metavar="UID.KEY=JSON",
        help="override a context_config field, e.g. pyannote-diarization.num_threads=2",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--service-log", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-hygiene", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.sessions < 1:
        raise SystemExit("--sessions must be at least 1")

    report = run_probe(args)
    write_json(Path(args.out), report)
    lat = report["latency"]
    print(f"\nWrote {args.out}")
    print(
        json.dumps(
            {
                "chunk_id_in_progress": lat["chunk_id_method"]["in_progress"],
                "chunk_id_final": lat["chunk_id_method"]["final"],
                "word_first_shown": lat["word_method"]["first_shown"],
                "label_after_text": lat["label_latency"]["after_text_shown"],
                "messages": lat["transcript_messages"],
                "speakers_updates": lat["speakers_update_messages"],
                "final_words": lat["final_words"],
                "labelled_fraction": lat["final_words_labelled_fraction"],
                "labels": lat["speaker_labels_seen"],
                "label_changes_before_final": lat["label_changes_before_final"],
                "label_changes_after_sent": lat["label_changes_after_sent"],
                "counters": report["service_counters_delta"],
                "diarization_cost": report["diarization_cost"],
                "workers": report["service"]["workers"],
                "rss_peak_mb": report["service"]["rss_mb_peak"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
