# Diarization benchmark

Measures whether candidate diarization engines keep up with ScribeAR's
5-second streaming job tick. Run it on the hardware that matters: the
CPU-only production target, and optionally a GPU machine (e.g. a university
cluster JupyterLab session) for comparison.

The decision this benchmark feeds: whether `pyannote-diarization` can be
enabled on CPU deployments (see `../../docs/speaker_diarization.md`), and
how it compares against NVIDIA Streaming Sortformer when a GPU is available.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install soundfile numpy torch torchaudio "pyannote.audio>=4.0,<5.0"
# Only needed for the sortformer engine (heavy install, GPU advised):
pip install "nemo_toolkit[asr]>=2.0"
```

Authenticate with HuggingFace (the pyannote pipeline is gated — accept its
terms at https://huggingface.co/pyannote/speaker-diarization-community-1):

```bash
export HUGGINGFACE_ACCESS_TOKEN=hf_...
```

## Test audio

Use a 3–10 minute multi-speaker recording (a lecture with questions, or a
two-person conversation). The repository's `test_audio_files` are synthetic
tones and contain no speech. Convert to the expected format:

```bash
ffmpeg -i recording.m4a -ac 1 -ar 16000 -acodec pcm_s16le sample.wav
```

## Run

```bash
python benchmark_diarization.py --audio sample.wav --engine pyannote --device cpu
python benchmark_diarization.py --audio sample.wav --engine both --device cuda
```

## Reading the results

- `Tick sim worst` is the number that matters: the slowest single
  re-diarization of the rolling 30s buffer. Live use requires it to fit
  well inside the 5s tick while faster-whisper shares the machine.
- CPU pyannote fitting the tick means production can enable diarization
  without any GPU dependency.
- If it does not fit, live diarization becomes a GPU-only feature flag and
  CPU deployments keep today's behavior.

Record results in `results.md` alongside the hardware used, so decisions
stay traceable.
