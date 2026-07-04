"""
Manual test client that streams a WAV file into a running transcription
service and prints transcripts (with speaker labels when present).

Start the service first (see transcription_service README / Makefile), then:

    uv run python tests/manual/transcription_stream_file_client.py \
        --audio sample.wav \
        --url ws://localhost:8000/transcription_stream/whisper \
        --api-key <API_KEY from .env>

The audio file must be 16 kHz mono. Convert with:
    ffmpeg -i input.m4a -ac 1 -ar 16000 -acodec pcm_s16le sample.wav

By default audio is sent in real time (1 second of audio per second) to
mimic a live microphone; pass --fast to stream without pacing.
"""

import argparse
import asyncio
import io
import json
import sys

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """
    Load a 16 kHz mono WAV file as float32 samples

    Args:
        path    - Path of audio file to load

    Returns:
        Mono float32 samples
    """
    samples, rate = sf.read(path, dtype="float32")
    if rate != SAMPLE_RATE:
        raise SystemExit(
            f"Expected {SAMPLE_RATE} Hz audio, got {rate} Hz. "
            "Convert with: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav"
        )
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def encode_wav_chunk(samples: np.ndarray) -> bytes:
    """
    Encode samples as a self-contained WAV file, as the service's
    AudioDecoder expects each binary message to be decodable on its own

    Args:
        samples - Mono float32 samples to encode

    Returns:
        WAV file bytes
    """
    buffer = io.BytesIO()
    sf.write(buffer, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def format_sequence(sequence: dict) -> str:
    """
    Render one transcript sequence, grouping words by speaker when the
    service provides speaker labels

    Args:
        sequence    - Transcript sequence message payload

    Returns:
        Human readable one-line rendering
    """
    text = sequence.get("text") or []
    speakers = sequence.get("speakers")
    if not speakers:
        return "".join(text)

    parts: list[str] = []
    current: str | None = None
    for word, speaker in zip(text, speakers):
        if speaker is not None and speaker != current:
            parts.append(f" [{speaker}]")
            current = speaker
        parts.append(word)
    return "".join(parts).strip()


async def print_transcripts(websocket) -> None:
    """
    Print transcript messages from the websocket until it closes

    Args:
        websocket   - Connected websocket to read from
    """
    async for message in websocket:
        if isinstance(message, bytes):
            continue
        payload = json.loads(message)
        if payload.get("type") != "transcript":
            print(f"<- {payload}")
            continue
        final = payload.get("final")
        in_progress = payload.get("in_progress")
        if final:
            print(f"FINAL      | {format_sequence(final)}")
        if in_progress:
            print(f"in progress| {format_sequence(in_progress)}")


async def stream_audio(
    websocket, samples: np.ndarray, chunk_sec: float, realtime: bool
) -> None:
    """
    Send audio to the websocket in chunk_sec sized WAV chunks

    Args:
        websocket   - Connected websocket to send audio on
        samples     - Full audio samples to stream
        chunk_sec   - Seconds of audio per chunk
        realtime    - Pace chunks at real time if True
    """
    chunk_samples = int(chunk_sec * SAMPLE_RATE)
    total_sec = samples.shape[0] / SAMPLE_RATE
    for start in range(0, samples.shape[0], chunk_samples):
        chunk = samples[start : start + chunk_samples]
        await websocket.send(encode_wav_chunk(chunk))
        sent_sec = min((start + chunk_samples) / SAMPLE_RATE, total_sec)
        print(f"-> sent {sent_sec:6.1f}s / {total_sec:.1f}s", file=sys.stderr)
        if realtime:
            await asyncio.sleep(chunk_sec)


async def run(args: argparse.Namespace) -> None:
    """
    Connect, authenticate, stream the file, and print transcripts

    Args:
        args    - Parsed command line arguments
    """
    samples = load_audio(args.audio)
    async with connect(args.url, max_size=None) as websocket:
        await websocket.send(
            json.dumps({"type": "auth", "api_key": args.api_key})
        )
        await websocket.send(json.dumps({"type": "config", "config": {}}))

        printer = asyncio.create_task(print_transcripts(websocket))
        await stream_audio(websocket, samples, args.chunk_sec, not args.fast)

        # Give the service time to flush transcripts for the audio tail
        await asyncio.sleep(args.linger_sec)
        printer.cancel()


def main() -> None:
    """
    Parse arguments and run the streaming client
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="16 kHz mono WAV file")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/transcription_stream/whisper",
        help="Transcription stream websocket URL (provider key in path)",
    )
    parser.add_argument(
        "--api-key", required=True, help="API_KEY configured for the service"
    )
    parser.add_argument(
        "--chunk-sec",
        type=float,
        default=1.0,
        help="Seconds of audio per websocket message",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Stream without real-time pacing"
    )
    parser.add_argument(
        "--linger-sec",
        type=float,
        default=15.0,
        help="Seconds to wait for trailing transcripts after streaming ends",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
