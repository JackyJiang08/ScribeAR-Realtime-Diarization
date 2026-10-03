"""
Warm-up model load for the benchmark suite.

Loads the three contexts the reference config uses (faster-whisper base,
Silero VAD, pyannote community-1) through the service's own context classes
and runs one short pass of each, so model downloads and first-load disk
reads happen here and never inside a timed run. Prints a JSON summary.

Usage (from transcription_service/):
  uv run python benchmarks/diarization/warmup.py [--no-pyannote]
"""

# pylint: disable=import-outside-toplevel,missing-function-docstring
# pylint: disable=broad-exception-caught

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_common import ROOT, SAMPLE_RATE, ensure_hf_token_env  # noqa: E402

sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pyannote", action="store_true")
    parser.add_argument("--whisper-model", default="base")
    args = parser.parse_args()

    from src.shared.logger import ContextLogger
    from src.transcription_contexts.faster_whisper_context import (
        FasterWhisperContext,
    )
    from src.transcription_contexts.silero_vad_context import SileroVadContext

    logging.basicConfig(level=logging.WARNING)
    log = ContextLogger(logging.getLogger("warmup"))
    token_var = ensure_hf_token_env()
    audio = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    timings: dict = {}

    started = time.perf_counter()
    whisper = FasterWhisperContext(
        {"model": args.whisper_model, "device": "cpu"}, ["w"]
    ).create(log)
    timings["whisper_load_sec"] = round(time.perf_counter() - started, 2)
    started = time.perf_counter()
    parts, _ = whisper.transcribe(audio, word_timestamps=True, language="en")
    list(parts)
    timings["whisper_pass_sec"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    SileroVadContext({}, ["s"]).create(log)
    timings["silero_load_sec"] = round(time.perf_counter() - started, 2)

    if not args.no_pyannote:
        from src.transcription_contexts.pyannote_diarization_context import (
            PyannoteDiarizationContext,
        )

        started = time.perf_counter()
        service = PyannoteDiarizationContext(
            {"device": "cpu", "token_env_var": token_var}, ["p"]
        ).create(log)
        timings["pyannote_load_sec"] = round(time.perf_counter() - started, 2)
        started = time.perf_counter()
        service.diarize(audio, SAMPLE_RATE)
        timings["pyannote_pass_sec"] = round(time.perf_counter() - started, 2)

    print(json.dumps(timings))


if __name__ == "__main__":
    main()
