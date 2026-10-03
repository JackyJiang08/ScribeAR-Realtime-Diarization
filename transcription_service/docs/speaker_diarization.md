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

3. Accuracy and performance on target hardware: `benchmarks/diarization/`
   ("Evaluation harness" below).

## Evaluation harness

Every Phase 2 change is judged by the harness in `benchmarks/diarization/`
(see its `README.md` for the file map). It produces one JSON report per run
with a flat `key_metrics` block, and a regression gate compares that block
against the committed baseline for the environment the run was made in.

### Reference configuration

All runs use `benchmarks/diarization/configs/reference_provider_config.json`:
upstream's shipped deployment config for the whisper-streaming provider
(`deployment/provider_config.template.json`: faster-whisper `base` on CPU,
Silero context registered, **no `vad_detector`**, 5 s job period, 30 s
buffer, `local_agree_dim` 2) with only the diarization settings added on
top. For "diarization off" the harness sets `diarization_detector: false`
and removes the pyannote context, so the off run is exactly upstream as
shipped. The fork's old development config (`vad_detector: true`, which the
production audit identified as the cause of the 27 to 30 s caption
latency) is kept as `configs/dev_vad_provider_config.json` and only runs
when asked for (`SUITE_ARGS=--with-dev-vad`); its results are labelled
`secondary_dev_vad` and never gated.

### Environments

| environment | how | baseline file |
|---|---|---|
| native (quick dev loop) | `make benchmark_diarization_suite` | `baselines/native-<os>-<arch>.json` |
| **Linux CPU reference** | `make benchmark_diarization_suite_docker` | `baselines/linux-cpu-4c8g.json` |

Production runs Linux, and some effects only show up there (Silero's
`torch.set_num_threads(1)` applying to pyannote, `OMP_NUM_THREADS=1` in the
image). The Docker mode builds upstream's `Dockerfile_CPU` image unchanged,
adds the pyannote extra and the benchmark sources
(`benchmarks/diarization/docker/Dockerfile`), and runs with fixed limits:
`BENCH_CPUS` (default 4) and `BENCH_MEMORY_GB` (default 8). The limits, the
cgroup values actually in force and the runner are recorded in every report
under `resource_limits`. Audio data, results and the HuggingFace and torch
caches are bind-mounted from the host. Docker Desktop's VM must have at least
as much memory as the limit, or the script warns that the limit is nominal.

### Run hygiene

Before a run the suite records free memory, swap in use and load averages
(`hygiene.before`) and again at the end. It warns on the console and marks
the report (`hygiene.clean: false`, `hygiene.warnings`) when swap is in use,
when the one-minute load exceeds 75 percent of the CPUs the run may use, or
when less than 3 GB is free. The gate prints a note when either side was
taken with warnings. A warm-up step loads whisper, Silero and pyannote once
and runs a short pass before anything is timed, so first-load cost and model
downloads never land in a measurement; the replay benchmark additionally
runs a warm-up pass on its own model instance.

### What the suite runs

`make benchmark_diarization_suite` (or `_docker`) runs, each in its own
process:

1. **Replay benchmark** (`benchmark_baseline.py`) over the AMI set
   (ES2004a, IS1009a, TS3003a, single distant microphone, first 10 min):
   one offline pass per file and a streaming replay of the job loop (5 s
   tick, 30 s buffer, `SpeakerReconciler`) over the first `STREAM_SEC`
   seconds (default 120; `STREAM_SEC=0` replays the full length, which is
   the option to switch to once the tick fits the period).
2. **Caption latency** (`caption_latency.py`): starts the real service with
   the reference config, streams ES2004a at real time in 0.5 s SAFP frames
   for `CAPTION_SEC` seconds (default 180), diarization **on** and **off**.

### Metrics

Per file and aggregated, in `report.replay`:

- `offline` / `streaming_first_seen` / `streaming_settled`: **DER** with
  its missed / false-alarm / confusion breakdown (pyannote convention: no
  collar, overlap scored; a 0.25 s collar variant is included), **JER**,
  and the **speaker count error** (hypothesis labels minus reference
  speakers). *First seen* is the label the newest tick gives in-progress
  words; *settled* is the label a region has two ticks later, which is what
  finalized words get.
- `label_latency_sec` p50 / p95: time from a reference speech onset until
  the first tick whose hypothesis covers it, including that tick's compute.
- `label_flip_rate_after_first_shown`: fraction of 5 s regions whose
  dominant label changed after a viewer first saw it (plus the older
  `label_flips_per_min`).
- `labels_minted_per_reference_speaker`: session labels the reconciler
  created divided by the people actually speaking in the streamed part.
- `tick_cost_sec` (mean / p50 / p95 / worst against the 5 s budget),
  `stage_cost_sec` per pyannote stage (segmentation, embeddings,
  clustering and other) and `reconciler_cost_sec`.
- `schedule`: the **lag behind real time** and the **ticks skipped**
  a real worker would show with these tick costs (the worker pool drops
  every period that elapses while a pass is still running). Modelled from
  the replay; the live counterpart is the `dropped_periods` counter in the
  caption-latency run.
- `memory`: RSS after the first streaming tick and at the end of each
  file's replay, its growth between those two points, min and max during
  the replay, plus the process peak. Over a 120 s replay the growth mostly
  reflects the rolling window filling from 5 s to 30 s, and in the native
  run the preceding offline pass is still releasing memory, so the native
  number can be negative; it is reported, not gated. Leak detection is the
  soak harness's job (below).

In `report.caption_latency.reference.{on,off}`:

- **chunk-id method (primary)**: delay from the send time of the newest
  chunk in `in_progress_chunk_ids` / `final_chunk_ids` to that message's
  arrival, p50 / p95. This is what node-server aggregates into its latency
  percentiles, so the numbers match the fleet dashboard.
- **word method (secondary)**: per `(end, text)` pair, delay from the chunk
  containing `end` to the first message showing the pair (`first_shown`)
  and to the first `final` showing it (`finalized`).
- service counters over the run: dropped periods, audio dropped because the
  buffer was full (count and seconds), buffer overflows, jobs completed; the
  execution-time histogram at the end; transcript messages, finalized
  words, labelled fraction, labels seen, label changes before final.
- service process-tree RSS after readiness, at the end, and its peak.

### Regression gate

`make benchmark_diarization_gate` (or `_docker`) runs the suite and then
`compare_baseline.py`, which picks `baselines/<environment.baseline_key>.json`
and fails (exit 1) when any gated metric is worse than the baseline beyond
`max(abs, rel * |baseline|)` from `baselines/gate_rules.json`. Accuracy
metrics carry an absolute tolerance of 0.02, timing metrics 15 to 20
percent, counts a small absolute margin; metrics without a rule are printed
for information only. To move the baseline after an accepted change, run
the suite and copy its report over the baseline file for that environment
(`RESULT=benchmarks/diarization/baselines/linux-cpu-4c8g.json make
benchmark_diarization_suite_docker`), then commit it.

The committed baselines are the official Phase 2 starting point measured on
the unmodified pipeline (commit recorded in each file's `code_revision`).

### Hard cases

`make benchmark_diarization_hardcases` prepares and replays the six cases in
`benchmarks/diarization/hard_cases.json`, each streamed in full:

| case | source | why |
|---|---|---|
| `overlap` | IS1009a 480 to 600 s | highest fraction of overlapping speech (26 percent) |
| `short_turns` | TS3003a 1320 to 1440 s | 24 reference turns under 1 s in two minutes, little overlap |
| `return_after_gap` | ES2004a 23 to 336 s | speaker FEE016 silent for 253 s (well past the 30 s buffer) and returning |
| `four_speakers` | ES2004a 750 to 870 s | all four speakers with at least 25 s of speech each in two minutes |
| `noise_clean` | TS3003a 1320 to 1440 s | clean counterpart for the noise case |
| `noise_pink_snr5` | same, plus synthesised pink noise at 5 dB SNR (seed 20261002) | background noise |

Audio comes from the AMI corpus mirror (single distant microphone
`Array1-01`, CC BY 4.0) and references from pyannote's AMI-diarization-setup
`only_words` RTTMs; the windows were chosen from the reference RTTMs alone by
`select_hard_cases.py` (the criteria are in its docstring and the manifest's
`selection` fields). Nothing under `data/` is ever committed.

### Long-session soak

`make benchmark_diarization_soak` (`SOAK_MINUTES=120` for two hours)
concatenates consecutive AMI meetings of the same four people (ES2004a to d,
then IS1009a to d when more is needed) into one stream with a matching RTTM
and replays it through the job loop. Per 5 min bin it reports DER and
confusion (drift), the session label covering most of each reference
speaker's speech and the **label swaps** between bins, labels minted so far,
tick cost and RSS; overall it reports swaps per hour, labels per reference
speaker, DER drift from the first to the last bin and memory growth. The
replay runs at compute speed: with today's 12 s ticks an hour of audio takes
about 2.5 hours, so run it on the reference environment once the tick fits
the period.

### Regression tests from the audit

`tests/unit/shared/utils/speaker_reconciler/speaker_reconciler_regressions_test.py`
turns the audit's synthetic reconciler cases (window sliding past a voice,
speaker returning after a long gap, backchannels minting labels, split and
merge ping-pong, merged speakers splitting again, reconnect restarting at
`spk_0`) into unit tests marked `xfail(strict=True)`. They document the
behaviour the Phase 2b fix must deliver and will fail the build the moment
one starts passing unexpectedly, so the markers come off as the fix lands.

### Requirements

`ffmpeg`, `HUGGINGFACE_ACCESS_TOKEN` (or `HF_TOKEN`) with the pyannote model
terms accepted, the `pyannote-diarization` extra installed
(`make install_dev_cpu`), and Docker for the reference environment.

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
3. **Evaluation suite.** `make benchmark_diarization_gate` (see
   "Evaluation harness" above) runs the standard suite and compares it
   with the committed baseline for this environment.
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
| Tooling | Manual end-to-end client, evaluation harness (replay benchmark, caption-latency probe, Linux CPU reference container, regression gate, hard cases, soak), this document |

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
