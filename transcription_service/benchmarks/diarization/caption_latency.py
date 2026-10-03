"""
End-to-end caption latency probe: runs the real transcription service with
a given provider config, streams a recording at it in real time as SAFP
frames (the same instrument shape as upstream's tools/asr-load and the
production kiosk) and measures how long captions take to appear.

Two latency methods are reported, both from the audit
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

It also records the service's own counters over the run (dropped periods,
audio dropped because the buffer was full, buffer overflows, execution-time
histograms) via /metrics/status, and the peak and final RSS of the service
process tree.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/caption_latency.py \
      --provider-config benchmarks/diarization/configs/reference_provider_config.json \
      --audio benchmarks/diarization/data/ami/ES2004a_10min.wav \
      --diarization on --seconds 180 --out results/caption_on.json
"""

# pylint: disable=too-many-locals,too-many-statements,too-many-instance-attributes
# pylint: disable=missing-function-docstring,broad-exception-caught
# pylint: disable=too-many-branches,import-outside-toplevel

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
    process_tree_rss_mb,
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
)
HISTOGRAMS_OF_INTEREST = ("asrExecutionMs", "asrSchedulingDelayMs", "asrRtf")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def prepare_config(path: Path, diarization_on: bool, provider: str) -> dict:
    """
    Loads the reference provider config and derives the diarization-off
    variant: `diarization_detector` false and the pyannote context removed,
    so the off run is exactly upstream's shipped configuration (no pyannote
    loaded, no HuggingFace token needed).
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
        provider_config.pop("diarization_min_speakers", None)
        provider_config.pop("diarization_max_speakers", None)
        config["contexts"] = [
            ctx
            for ctx in config["contexts"]
            if ctx["context_uid"] != "pyannote-diarization"
        ]
    return config


def encode_wav_chunk(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class ServiceProcess:
    """The transcription service as a child process with an RSS sampler."""

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
        self.rss_detail_last: dict = {}
        self._stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()

    def _sample(self):
        while not self._stop.is_set():
            try:
                reading = process_tree_rss_mb(self.process.pid)
                self.rss_samples.append(reading["total_mb"])
                self.rss_detail_last = reading
            except Exception:
                pass
            self._stop.wait(1.0)

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
    """Correlates transcript messages with the send times of audio chunks."""

    def __init__(self, chunk_sec: float):
        self.chunk_sec = chunk_sec
        self.send_time: dict[int, float] = {}
        self.chunk_id_in_progress: list[float] = []
        self.chunk_id_final: list[float] = []
        self.word_first_shown: dict[tuple, float] = {}
        self.word_finalized: dict[tuple, float] = {}
        self.word_first_speaker: dict[tuple, str | None] = {}
        self.label_changes_before_final = 0
        self.messages = 0
        self.messages_with_text = 0
        self.final_words = 0
        self.final_words_labelled = 0
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

    def on_message(self, payload: dict, arrival: float):
        if payload.get("type") != "transcript":
            return
        self.messages += 1
        if self.first_message_at is None:
            self.first_message_at = arrival
        for kind, key in (
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
                        if kind == "in_progress"
                        else self.chunk_id_final
                    ).append(latency)
        had_text = False
        for kind in ("in_progress", "final"):
            sequence = payload.get(kind)
            if not sequence or not sequence.get("text"):
                continue
            had_text = True
            texts = sequence["text"]
            ends = sequence.get("ends") or [None] * len(texts)
            speakers = sequence.get("speakers") or [None] * len(texts)
            for text, end, speaker in zip(texts, ends, speakers):
                if end is None:
                    continue
                word_key = (round(float(end), 2), text)
                chunk_index = min(
                    int(float(end) // self.chunk_sec), self.last_sent_index
                )
                if chunk_index not in self.send_time:
                    continue
                latency = arrival - self.send_time[chunk_index]
                if word_key not in self.word_first_shown:
                    self.word_first_shown[word_key] = latency
                    self.word_first_speaker[word_key] = speaker
                if kind == "final":
                    if word_key not in self.word_finalized:
                        self.word_finalized[word_key] = latency
                        self.final_words += 1
                        if speaker is not None:
                            self.final_words_labelled += 1
                            self.labels_seen.add(speaker)
                        first = self.word_first_speaker.get(word_key)
                        if (
                            first is not None
                            and speaker is not None
                            and first != speaker
                        ):
                            self.label_changes_before_final += 1
        if had_text:
            self.messages_with_text += 1

    def report(self) -> dict:
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
            "transcript_messages": self.messages,
            "transcript_messages_with_text": self.messages_with_text,
            "final_words": self.final_words,
            "final_words_labelled": self.final_words_labelled,
            "final_words_labelled_fraction": (
                round(self.final_words_labelled / self.final_words, 3)
                if self.final_words
                else None
            ),
            "speaker_labels_seen": sorted(self.labels_seen),
            "label_changes_before_final": self.label_changes_before_final,
        }


async def stream(
    url: str,
    samples: np.ndarray,
    chunk_sec: float,
    seconds: float,
    linger_sec: float,
    collector: LatencyCollector,
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
                    "session_uid": "diarization-benchmark",
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
            if index % 20 == 0:
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


def run_probe(args) -> dict:
    token_var = ensure_hf_token_env()
    diarization_on = args.diarization == "on"
    config = prepare_config(
        Path(args.provider_config), diarization_on, args.provider
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
        f"starting service (diarization {args.diarization}) on port {port}",
        file=sys.stderr,
        flush=True,
    )
    service = ServiceProcess(config_path, port, log_path, env_extra)
    try:
        ready_sec = service.wait_ready(READINESS_TIMEOUT_SEC)
        print(
            f"service ready after {ready_sec:.1f}s", file=sys.stderr, flush=True
        )
        rss_ready = process_tree_rss_mb(service.process.pid)
        metrics_before = service.metrics()
        collector = LatencyCollector(chunk_sec)
        url = f"ws://127.0.0.1:{port}/transcription_stream/{args.provider}"
        wall_start = time.perf_counter()
        schedule = asyncio.run(
            stream(
                url,
                samples,
                chunk_sec,
                args.seconds,
                args.linger_sec,
                collector,
            )
        )
        wall = time.perf_counter() - wall_start
        metrics_after = service.metrics()
        rss_end = process_tree_rss_mb(service.process.pid)
        exited = service.process.poll()
    finally:
        service.stop()

    before = summarize_counters(metrics_before)
    after = summarize_counters(metrics_after)
    latency = collector.report()
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
            "effective_provider_config": config["providers"][args.provider][
                "provider_config"
            ],
            "contexts": [ctx["context_uid"] for ctx in config["contexts"]],
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
        },
        "stream": schedule | {"wall_sec": round(wall, 1)},
        "latency": latency,
        "service_counters_delta": counter_deltas(before, after),
        "service_histograms_end": summarize_histograms(metrics_after),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-config", required=True)
    parser.add_argument("--provider", default="whisper")
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV")
    parser.add_argument("--diarization", choices=["on", "off"], default="on")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--linger-sec", type=float, default=20.0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--service-log", default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-hygiene", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
                "word_finalized": lat["word_method"]["finalized"],
                "messages": lat["transcript_messages"],
                "final_words": lat["final_words"],
                "labels": lat["speaker_labels_seen"],
                "counters": report["service_counters_delta"],
                "rss_peak_mb": report["service"]["rss_mb_peak"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
