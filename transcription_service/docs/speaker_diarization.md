# Speaker diarization

The whisper-streaming provider can optionally label each transcribed word
with a speaker (`spk_0`, `spk_1`, ...), so clients can render who said what.
Diarization runs on CPU by default and is fully optional: deployments that do
not configure it behave exactly as before.

## How it works

Each job tick the provider already re-transcribes the rolling audio buffer.
With diarization enabled it additionally:

1. Runs a pyannote speaker diarization pipeline over the same buffer,
   producing raw per-run speaker ranges (`PyannoteDiarizationContext`).
2. Maps the raw labels to stable session-wide labels. Diarization models
   assign arbitrary labels per run, so the same voice could flip between
   `SPEAKER_00` and `SPEAKER_01` on consecutive ticks; `SpeakerReconciler`
   fixes this by matching each run's labels to the previous run's labels
   through overlap voting on the shared window.
3. Assigns each transcribed word the speaker range it overlaps the most.
4. Emits labels through the existing `TranscriptionSequence` as a `speakers`
   array aligned with `text`. Words with no attributable speaker carry `null`.

Speaker labels do not participate in Local Agreement finalization: a label
flip never delays caption finalization.

## Enabling it

1. Install the optional dependency group:

   ```bash
   uv sync --extra pyannote-diarization
   ```

2. Accept the model terms on HuggingFace
   (https://huggingface.co/pyannote/speaker-diarization-community-1) and set
   the access token in the service environment:

   ```bash
   export HUGGINGFACE_ACCESS_TOKEN=hf_...
   ```

3. Add the context and enable the provider flag in your provider config
   (see `provider_config.template.json`):

   ```json
   {
       "context_uid": "pyannote-diarization",
       "worker_ids": [0],
       "tags": ["pyannote_diarization"],
       "context_config": {
           "model": "pyannote/speaker-diarization-community-1",
           "device": "cpu",
           "token_env_var": "HUGGINGFACE_ACCESS_TOKEN"
       }
   }
   ```

   and in the whisper-streaming `provider_config`:

   ```json
   "diarization_detector": true,
   "diarization_context_tag": "pyannote_diarization"
   ```

   Optional bounds: `diarization_min_speakers`, `diarization_max_speakers`
   (a lecture with occasional questions usually works well with
   `diarization_max_speakers: 4`).

## Testing

1. Unit tests (no models needed):

   ```bash
   make install_dev_cpu
   make test_unit
   ```

2. End-to-end against a locally running service. Create a `.env`
   (see repository docs) pointing `PROVIDER_CONFIG_PATH` at a config with
   `diarization_detector: true`, export `HUGGINGFACE_ACCESS_TOKEN`, start
   the service with `make dev`, then stream a recording into it:

   ```bash
   uv run python tests/manual/transcription_stream_file_client.py \
       --audio sample.wav --api-key <API_KEY from .env>
   ```

   Finalized lines print with inline `[spk_N]` markers on speaker changes.
   Use a 16 kHz mono WAV with at least two speakers; the first diarization
   run downloads the pyannote model, so expect a slow first tick.

3. Performance measurement on target hardware: `benchmarks/diarization/`.

## Accuracy and speed baseline

`make benchmark_diarization_baseline` reproduces the reference benchmark in
one command. It downloads three AMI corpus meetings (ES2004a, IS1009a,
TS3003a; single distant microphone, first 10 minutes) with the standard
pyannote `only_words` reference RTTMs into the gitignored
`benchmarks/diarization/data/` folder, then runs
`benchmarks/diarization/benchmark_baseline.py` on CPU and writes a JSON
report to `benchmarks/diarization/results/`.

The report contains, per file and aggregated:

- **offline** DER (pyannote convention: no collar, overlap scored) with its
  missed-speech / false-alarm / confusion breakdown, a 0.25 s collar
  variant, JER, real-time factor and peak memory.
- **streaming** replay of the job loop (5 s tick, 30 s rolling buffer,
  `SpeakerReconciler` labels): DER for the label a viewer sees first and for
  the label a region settles on two ticks later, per-tick cost against the
  5 s budget, label latency from speech onset, label flips per minute and
  how many session labels were minted. The replay covers the first
  `STREAM_SEC` seconds of each file (default 120) because each tick
  re-diarizes up to 30 s of audio.

Requirements: `ffmpeg`, `HUGGINGFACE_ACCESS_TOKEN` (or `HF_TOKEN`) with the
pyannote model terms accepted, and the `pyannote-diarization` extra
installed. `make benchmark_diarization_baseline RESULT=... STREAM_SEC=...`
overrides the report path and replay length. The committed
`results/pre_sync_baseline.json` is the reference run from before the
upstream sync; compare new runs against it.

## Performance notes

Diarization shares the worker's CPU budget with Whisper. If ticks start
exceeding `job_period_ms`, pin the diarization context to its own worker
process: raise `num_workers` and give the `pyannote-diarization` context a
dedicated entry in `worker_ids`. Measure real hardware with
`benchmarks/diarization/` before enabling in production.

## Fork testing checklist

Everything the fork adds can be verified from a clean clone of
`JackyJiang08/ScribeAR-Realtime-Diarization` on `feature/speaker-diarization`.

1. **Unit tests, no models needed.** Python (3.12 + uv):
   `make install_dev_cpu`, `make format`, `make lint` (must score 10/10),
   `make test_unit`. TypeScript (Node 20+): `npm ci`, `npm run build`,
   `npm run lint`, `npm run test:unit`, which includes the speaker-run
   grouping, reducer and WCAG color-contrast tests. Two
   `worker_process_manager` timing tests are known to be flaky on loaded
   laptops and are unrelated to diarization.
2. **End to end with real audio.** Follow "Testing" above: 16 kHz mono WAV
   with at least two speakers, token exported, `diarization_detector: true`,
   `make dev`, then stream the file with the manual client. Finalized lines
   carry inline `[spk_N]` markers; the same voice must keep the same label
   for the whole session, which is the `SpeakerReconciler` working.
3. **Accuracy and speed baseline.** `make benchmark_diarization_baseline`
   (see above). Compare against `results/pre_sync_baseline.json`.
4. **Full-stack UI check.** Run the Docker Compose stack in `deployment/`,
   join a session from the client webapp and speak with two people: captions
   show `Speaker 1:` / `Speaker 2:` labels in distinct colors that stay
   readable (WCAG AA, contrast at least 4.5:1) on any background theme.

## What the fork adds, by layer

| Layer | Change |
|---|---|
| `transcription_service` | `PyannoteDiarizationContext`: optional worker-pool context running `pyannote/speaker-diarization-community-1` (CPU by default, CUDA-ready) |
| `transcription_service` | `SpeakerReconciler`: maps per-run labels to session-wide labels by overlap voting, so one voice keeps one label across streaming re-runs |
| Wire format | Optional `speakers` array aligned with word tokens, end to end through the WebSocket messages and the shared TypeScript schemas; omitted entirely when diarization is off |
| Client UI | `Speaker N:` labels on speaker change, colored from a colorblind-aware palette auto-adjusted to WCAG AA contrast against the configured background |
| Tooling | Manual end-to-end client, CPU/GPU timing harness, AMI accuracy baseline, this document |

Planned follow-ups beyond diarization: resumable lecture summarization
(prompt-injection resistant) and RNNoise-based denoising.

This fork is an independent feature-development copy built on ScribeAR by the
ScribeAR team at the University of Illinois Urbana-Champaign; upstream retains
all rights to the original code.

## Known limitations

- Labels can still drift after long silences or when the rolling buffer is
  force-purged past the last diarized window; the reconciler then mints a
  fresh label for what may be the same voice. Fixing this properly requires
  speaker-embedding memory, which is planned as a follow-up.
- Words already finalized keep the label they were finalized with; later,
  better speaker evidence does not retroactively update them.
- Accuracy degrades with heavily overlapping speech and far-field audio.
- GPU deployments can set `"device": "cuda"` in the context config; a
  streaming-native alternative (NVIDIA Streaming Sortformer) can be added as
  another context implementation without touching the provider wiring.
