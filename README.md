# ScribeAR — Real-Time Speaker Diarization

Feature-development repository for [ScribeAR](https://github.com/scribear/scribear),
UIUC's real-time classroom captioning system for accessibility. This repository
extends ScribeAR with **live speaker diarization**: captions are labeled with
stable speaker identities (`Speaker 1`, `Speaker 2`, ...) as people talk, with
colorblind-aware, WCAG-AA-contrast speaker colors in the client UI.

All feature work (including this README) lives on the
**`feature/speaker-diarization`** branch; `staging` mirrors the upstream
baseline. Planned follow-ups: resumable lecture summarization
(prompt-injection-resistant) and RNNoise-based denoising.

## What this branch adds

| Layer | Change |
|---|---|
| `transcription_service` (Python) | `PyannoteDiarizationContext` — optional worker-pool context running `pyannote/speaker-diarization-community-1` (CPU by default, CUDA-ready) |
| `transcription_service` (Python) | `SpeakerReconciler` — stabilizes per-run diarization labels into session-wide labels via overlap voting, so a speaker keeps one label across streaming re-runs |
| Wire format | Optional `speakers` array aligned with word tokens, end to end through the WebSocket messages and shared TypeScript schemas; omitted entirely when diarization is off |
| Client UI (React) | `Speaker N:` labels on speaker change, colored from a colorblind-aware palette that is auto-adjusted to meet WCAG AA contrast (≥ 4.5:1) against the user-configured background |
| Tooling | End-to-end manual test client, CPU/GPU benchmark harness, docs |

Speaker labels never participate in caption finalization (Local Agreement), so
label changes cannot delay live captions. Deployments that do not enable
diarization behave byte-for-byte as before.

Design notes and configuration reference:
[`transcription_service/docs/speaker_diarization.md`](transcription_service/docs/speaker_diarization.md).

## Testing

All commands below run on this branch:

```bash
git clone https://github.com/JackyJiang08/scribear-realtime-diarization.git
cd scribear-realtime-diarization
git checkout feature/speaker-diarization
```

### 1. Unit tests (fast, no ML models required)

Python service (requires Python 3.12 and [uv](https://docs.astral.sh/uv/)):

```bash
cd transcription_service
make install_dev_cpu   # installs deps incl. CPU torch, faster-whisper, pyannote
make format            # isort + black checks
make lint              # pylint, must score 10/10
make test_unit         # pytest with coverage
```

TypeScript monorepo (requires Node 20+):

```bash
npm ci
npm run build          # type-checks and builds every workspace
npm run lint
npm run test:unit      # includes speaker-run grouping, reducer, and
                       # WCAG color-contrast property tests
```

Expected: all suites pass. (Two `worker_process_manager` timing tests are
known to be flaky on loaded laptops; they are unrelated to diarization.)

### 2. End-to-end test with real audio

This streams a recording into a locally running transcription service exactly
like a classroom microphone would, and prints speaker-labeled transcripts.

1. **Prepare a recording** — 3–10 minutes with at least two speakers works
   best. Convert it to 16 kHz mono WAV:

   ```bash
   ffmpeg -i recording.m4a -ac 1 -ar 16000 -acodec pcm_s16le sample.wav
   ```

2. **Get model access** — accept the terms of the gated pyannote pipeline at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>, then:

   ```bash
   export HUGGINGFACE_ACCESS_TOKEN=hf_...
   ```

3. **Configure the service** — in `transcription_service/`, copy
   `provider_config.template.json` to `provider_config.json` and set
   `"diarization_detector": true` in the whisper provider config. Create a
   `.env`:

   ```env
   PORT=8000
   HOST=0.0.0.0
   API_KEY=dev-secret
   WS_INIT_TIMEOUT_SEC=5
   PROVIDER_CONFIG_PATH=./provider_config.json
   ```

4. **Run the service**:

   ```bash
   make dev
   ```

5. **Stream the recording** (second terminal):

   ```bash
   uv run python tests/manual/transcription_stream_file_client.py \
       --audio sample.wav --api-key dev-secret
   ```

Expected output: `FINAL` lines with inline `[spk_N]` markers at speaker
changes, e.g.

```
FINAL      | [spk_0] Hello everyone, welcome to class. [spk_1] Professor, I have a question.
```

Verify that the **same voice keeps the same label for the whole session** —
that is the `SpeakerReconciler` working. The first tick is slow while the
pyannote model downloads.

### 3. Performance benchmark on target hardware

Live use requires each re-diarization of the rolling 30 s buffer to fit inside
the 5 s streaming job tick. Measure on the machine that will run production:

```bash
cd transcription_service/benchmarks/diarization
python3 -m venv .venv && source .venv/bin/activate
pip install soundfile numpy torch torchaudio "pyannote.audio>=4.0,<5.0"
python benchmark_diarization.py --audio sample.wav --engine pyannote --device cpu
```

Read the `Tick sim worst` number: it must stay well under 5 s while
faster-whisper shares the machine. See
[`transcription_service/benchmarks/diarization/README.md`](transcription_service/benchmarks/diarization/README.md)
for GPU comparison runs (NVIDIA Streaming Sortformer).

### 4. Full-stack UI check (optional)

Run the complete stack with Docker Compose (see `deployment/`), join a session
from the client webapp, and speak with two people: captions should show
`Speaker 1:` / `Speaker 2:` labels in distinct, readable colors on any
background theme.

## Acknowledgments

Built on [ScribeAR](https://github.com/scribear/scribear) by the ScribeAR team
at the University of Illinois Urbana-Champaign. This repository is an
independent feature-development copy; upstream retains all rights to the
original code.
