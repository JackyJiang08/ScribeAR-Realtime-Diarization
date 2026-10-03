# Speaker diarization

The whisper-streaming provider can optionally label each transcribed word
with a speaker (`spk_0`, `spk_1`, ...), so clients can render who said what.
Diarization runs on CPU by default and is fully optional: deployments that do
not configure it register exactly the job upstream does and behave exactly
as before.

Since Phase 2a (2026-10-03) **captions never wait for diarization**. The two
run as separate worker-pool jobs on separate worker processes; captions are
emitted the moment Whisper produces them, and speaker labels follow as a
separate, backward-compatible message.

## How it works

Per session the provider registers two worker-pool jobs, fed the same audio
chunks:

1. **The caption job** (`WhisperStreamingProviderJob`) is upstream's job,
   byte for byte: Silero and Whisper on worker 0, Local Agreement
   finalization, the same results, counters and chunk ids as before. It
   knows nothing about speakers.
2. **The diarization job** (`DiarizationJob`) runs on whichever worker owns
   the `pyannote_diarization` context (worker 1 in the shipped configs),
   once per `diarization_period_ms`. Each pass it diarizes only the newest
   `diarization_window_sec` of audio (10 s by default: see "Choosing the
   window" for why), maps pyannote's per-pass labels onto stable session
   labels with `SpeakerReconciler` (overlap voting against the previous
   pass; consecutive windows overlap by window minus period, which is what
   carries identity across), and reports the labelled segments together
   with the window it covered and how old the audio was when the labels
   were ready (the **diarization lag**).

The session (main process) joins the two with `SpeakerLabelAttacher`:

- Every caption result is forwarded immediately. Words whose audio
  diarization has already covered carry their label in the `speakers`
  array; the rest carry `null`. A finalized sequence is sent with a
  session-unique `sequence_id`.
- Every diarization result extends an append-only, frozen label timeline up
  to `window_end - diarization_edge_margin_sec` (the last half second of a
  window is left for the next pass, which sees it with context). A label
  already on the timeline is never rewritten, so a word's label is decided
  exactly once, by the first pass that covered it, and the same label is
  what the in-progress tail, the finalized sequence and any later update
  show. Finalized words that were still `null` are sent later as a
  `speakers_update` naming the `sequence_id`; a client that ignores the
  message still has every caption.
- Audio no pass covered (the diarization job fell behind and skipped
  ahead) is decided as "no speaker" the moment the next pass reports its
  window, and a finalized sequence that has waited longer than
  `diarization_label_timeout_sec` (15 s) is settled with the labels it has.
  Nothing can leave a caption pending forever.

Back-pressure is by skipping, never by queueing or by slowing captions. The
diarization job's buffer holds exactly one window; audio that arrived while
a pass was running and no longer fits is purged unlabelled and counted
(`diarization_uncovered_seconds`), and a period the pass overran is dropped
by the worker pool and counted (`diarization_dropped_periods`), both exported
(see "Monitoring"). The pyannote context caps its torch threads (1 by
default) and lowers its worker's OS scheduling priority (`nice` 10), so on a
small machine the caption worker always wins a contended core: if CPU is
short, labels get slower, captions do not.

Whisper's word timestamps count only the audio its buffer kept, while the
diarization job counts every chunk it received, so the two clocks diverge
by the audio Whisper drops when its buffer is full (a service stall). The
session reads the caption job's `audio_dropped_buffer_full_seconds` counter
and shifts later word times by it before looking labels up.

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

3. Give the pyannote context **a worker of its own** and enable the provider
   flag (see `provider_config.template.json`):

   ```json
   "num_workers": 2,
   "contexts": [
       { "context_uid": "faster-whisper", "worker_ids": [0], ... },
       { "context_uid": "silero-vad",     "worker_ids": [0], ... },
       {
           "context_uid": "pyannote-diarization",
           "worker_ids": [1],
           "tags": ["pyannote_diarization"],
           "context_config": {
               "model": "pyannote/speaker-diarization-community-1",
               "device": "cpu",
               "token_env_var": "HUGGINGFACE_ACCESS_TOKEN",
               "num_threads": 1,
               "nice": 10,
               "segmentation_step": null
           }
       }
   ]
   ```

   and in the whisper-streaming `provider_config`:

   ```json
   "diarization_detector": true,
   "diarization_context_tag": "pyannote_diarization",
   "diarization_period_ms": 5000,
   "diarization_window_sec": 10,
   "diarization_edge_margin_sec": 0.5,
   "diarization_label_timeout_sec": 15
   ```

   Only the first two are required; the rest show their defaults. Optional
   bounds: `diarization_min_speakers`, `diarization_max_speakers` (a lecture
   with occasional questions usually works well with
   `diarization_max_speakers: 4`).

The provider fails at start-up if no live worker owns the diarization tag,
and warns if the only workers that do also run captions: a shared worker
runs one job at a time, so every diarization pass would hold captions up for
its duration and the context's `nice` would slow the caption job too. The
Docker images do not ship the pyannote extra yet (Phase 2c).

Context settings:

| field | default | meaning |
|---|---|---|
| `num_threads` | 1 | torch intra-op threads per pass. The embedding stage barely speeds up with more threads on the 4-CPU reference container while the caption worker slows down measurably; raise only with cores to spare. |
| `nice` | 10 | `os.nice` increment for the worker that loads this context; 0 disables. |
| `segmentation_step` | `null` (model default 0.1 = 1 s) | pyannote's segmentation step as a ratio of its 10 s window. Embedding windows per pass = (window - 10 s) / step + 1; irrelevant at a 10 s window, the second cost lever above it. |

## Wire format

`transcript` messages are unchanged in shape. With diarization on:

- `final` and `in_progress` sequences carry `speakers` (aligned with `text`,
  `null` = no label yet) and finalized sequences carry `sequence_id`
  (`"s0"`, `"s1"`, ... per session). With diarization off both fields are
  `null`, as they already were on this branch.
- A new server message attaches late labels:

  ```json
  {"type": "speakers_update", "sequence_id": "s3", "speakers": ["spk_0", null], "settled": false}
  ```

  `speakers` is aligned with that sequence's `text`; a `null` is a word
  without a label yet, or, once `settled` is true, a word that will never
  get one. An update only fills nulls: a label already sent for a word is
  never changed, by the service and by the clients alike.

node-server relays it as `speakersUpdate` (`sequenceId`, `speakers`,
`settled`) and carries `sequenceId` on transcript fragments; both are
`Type.Optional` in the shared schemas, so an older peer still validates.
Python: `SpeakersUpdateMessage`, `TranscriptSequence.sequence_id`;
TypeScript: `SPEAKERS_UPDATE_SCHEMA`, `TRANSCRIPT_FRAGMENT_SCHEMA`.

## Client UI

The content store (`@scribear/transcription-content-store`) keeps
`sequenceId` and `speakersSettled` on finalized sequences and applies
`applySpeakersUpdate` by filling nulls only. `isAwaitingSpeakers` tells a
renderer whether a sequence may still receive labels.

In the display (`@scribear/transcription-display-ui`), a sequence that came
from a diarizing provider (one carrying a `sequenceId`) always starts its
own line with a fixed-width **label slot** (`SpeakerLabelSlot`), decided the
moment the text appears: the slot shows `Speaker ?` while the label is
unknown and the label (`Speaker 2:`) once it arrives, in the same inline
element, so no line break is inserted and no text moves when the label fills
in. A sequence whose speaker continues from the previous line keeps the slot
but leaves it blank (a hanging indent), and a sequence the provider settled
without a label keeps `Speaker ?` as an explicit unattributed state. Speaker
changes inside one sequence are labelled inline, as before. The slot is
`aria-live="off"`, so a live region never announces the placeholder-to-label
change as a text update; the label is still read in order when a screen
reader walks the line. Sequences without a `sequenceId` (diarization off,
the standalone app) render exactly as before.

## Testing

1. Unit tests (no models needed):

   ```bash
   make install_dev_cpu
   make test_unit
   ```

2. End-to-end against a locally running service. Create a `.env`
   (see repository docs) pointing `PROVIDER_CONFIG_PATH` at a config with
   `diarization_detector: true` and the pyannote context on its own worker,
   export `HUGGINGFACE_ACCESS_TOKEN`, start the service with `make dev`,
   then stream a recording into it:

   ```bash
   uv run python tests/manual/transcription_stream_file_client.py \
       --audio sample.wav --api-key <API_KEY from .env>
   ```

   Finalized lines print with inline `[spk_N]` markers on speaker changes,
   and `speakers_update` messages are printed as they arrive. Use a 16 kHz
   mono WAV with at least two speakers; the model is downloaded when the
   worker starts, so expect a slow start the first time.

3. Accuracy and performance on target hardware: `benchmarks/diarization/`
   ("Evaluation harness" below).

## Monitoring

`/metrics/status` exports, keyed by the caption provider's `provider_key`
and all empty when diarization is off:

| counter | meaning |
|---|---|
| `diarizationRunsTotal`, `diarizationSecondsTotal` | passes and their wall time; seconds / runs is the per-pass cost |
| `diarizationFailedTotal` | passes that raised (captions unaffected, that audio unlabelled) |
| `reconcilerSecondsTotal`, `diarizationLabelsMintedTotal` | reconciler cost and session labels minted (far above the people in the room means drift) |
| `diarizationAudioSecondsTotal` | audio the job received, the RTF denominator |
| `diarizationUncoveredSecondsTotal` | audio no pass covered because the job fell behind: those words stay `Speaker ?` |
| `diarizationDroppedPeriodsTotal` | diarization periods skipped because the previous pass overran |

and histograms `diarizationExecutionMs`, `diarizationLagMs` (age of the
newest labelled audio when its labels were ready) and `diarizationRtf`. The
caption series (`asr*`) are untouched: the metrics registry folds the
diarization job's executions apart by its observer label.

The monitoring sidecar polls all of them (optional fields, so an older
service still validates), publishes `scribear_diarization_*` with a
`scribear_diarization_supported` guard, and raises
`diarizationBehindRule`: a warning when more than a tenth of the last
window's audio was skipped (`ALERT_DIARIZATION_UNCOVERED_RATIO`) or the p95
lag exceeds 10 s (`ALERT_DIARIZATION_LAG_P95_MS`), and a critical when audio
keeps arriving but no pass completes at all (the hung-job case, read from
the audio counter the job cannot censor). The Grafana fleet dashboard has
two diarization panels (pass cost and lag against the job period; audio
skipped, dropped periods and failed passes).

## Evaluation harness

Every Phase 2 change is judged by the harness in `benchmarks/diarization/`
(see its `README.md` for the file map). It produces one JSON report per run
with a flat `key_metrics` block, a regression gate compares that block
against the committed baseline for the environment the run was made in, and
`acceptance.py` checks the Phase 2a targets.

### Reference configuration

All runs use `benchmarks/diarization/configs/reference_provider_config.json`:
upstream's shipped deployment config for the whisper-streaming provider
(`deployment/provider_config.template.json`: faster-whisper `base` on CPU,
Silero context registered, **no `vad_detector`**, 5 s job period, 30 s
buffer, `local_agree_dim` 2) with only the diarization settings added on
top: a second worker owning the pyannote context, and the Phase 2a defaults
(10 s window, 5 s period, 1 torch thread, nice 10). For "diarization off"
the harness sets `diarization_detector: false`, removes the pyannote context
and its worker, so the off run is exactly upstream as shipped on one
worker. The fork's old development config (`vad_detector: true`, which the
production audit identified as the cause of the 27 to 30 s caption latency)
is kept as `configs/dev_vad_provider_config.json` and only runs when asked
for (`SUITE_ARGS=--with-dev-vad`); its results are labelled
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
**The Linux container run is the one that decides acceptance.**

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
   one offline pass per file and a streaming replay of the diarization job
   loop (the reference config's period and window, `SpeakerReconciler`)
   over the first `STREAM_SEC` seconds (default 120; `STREAM_SEC=0` replays
   the full length).
2. **Caption latency** (`caption_latency.py`): starts the real service with
   the reference config, streams ES2004a at real time in 0.5 s SAFP frames
   for `CAPTION_SEC` seconds (default 180), diarization **on** and **off**.
3. Optionally **concurrency** (`SESSIONS="2 3 4"` with
   `make benchmark_diarization_concurrency_docker`): the same stream over N
   concurrent sessions with diarization on.

### Metrics

Per file and aggregated, in `report.replay`:

- `offline` / `streaming_first_seen` / `streaming_settled`: **DER** with
  its missed / false-alarm / confusion breakdown (pyannote convention: no
  collar, overlap scored; a 0.25 s collar variant is included), **JER**,
  and the **speaker count error** (hypothesis labels minus reference
  speakers). *First seen* is the label the newest pass gives in-progress
  words; *settled* is the label a region has two passes later. With a 10 s
  window and a 5 s period a region is seen by two passes, so settled equals
  first seen; the service freezes the first label anyway (see "How it
  works"), so first seen is what a viewer gets.
- `label_latency_sec` p50 / p95: time from a reference speech onset until
  the first pass whose hypothesis covers it, including that pass's compute.
- `label_flip_rate_after_first_shown`: fraction of 5 s regions whose
  dominant label changed after a viewer first saw it (plus the older
  `label_flips_per_min`).
- `labels_minted_per_reference_speaker`: session labels the reconciler
  created divided by the people actually speaking in the streamed part.
- `tick_cost_sec` (mean / p50 / p95 / worst against the period budget),
  `stage_cost_sec` per pyannote stage (segmentation, embeddings,
  clustering and other) and `reconciler_cost_sec`.
- `schedule`: the **lag behind real time** and the **ticks skipped**
  a real worker would show with these tick costs (the worker pool drops
  every period that elapses while a pass is still running). Modelled from
  the replay; the live counterparts are `diarization_dropped_periods` and
  `diarization_uncovered_seconds` in the caption-latency run.
- `memory`: RSS after the first streaming tick and at the end of each
  file's replay, its growth between those two points, min and max during
  the replay, plus the process peak. Reported, not gated; leak detection is
  the soak harness's job.

In `report.caption_latency.reference.{on,off}`:

- **chunk-id method (primary)**: delay from the send time of the newest
  chunk in `in_progress_chunk_ids` / `final_chunk_ids` to that message's
  arrival, p50 / p95. This is what node-server aggregates into its latency
  percentiles, so the numbers match the fleet dashboard.
- **word method (secondary)**: per `(end, text)` pair, delay from the chunk
  containing `end` to the first message showing the pair (`first_shown`)
  and to the first `final` showing it (`finalized`).
- **label latency**: per finalized word, wall-clock delay from the message
  that first showed its text to the message that first gave it a non-null
  label (in-progress, final or `speakers_update`), `after_text_shown` and
  `after_finalized`, p50 / p95. Also the fraction of finalized words that
  ever got a label, `speakers_update` messages, sequences settled, label
  changes between first showing and finalization, and
  `label_changes_after_sent` (an update disagreeing with a sent label; zero
  by design).
- service counters over the run: dropped periods, audio dropped because the
  buffer was full (count and seconds), buffer overflows, jobs completed, the
  diarization counters; the execution-time, diarization-execution and lag
  histograms at the end.
- `service.workers`: every worker process of the service tree in spawn
  order (worker 0 captions, worker 1 diarization under the reference
  config) with its CPU cores over the streamed window and peak RSS; and
  `diarization_cost`: RTF from the service's own counters, pass cost, audio
  skipped, dropped periods, and the diarization worker's cores and RSS.

### Regression gate and acceptance

`make benchmark_diarization_gate` (or `_docker`) runs the suite and then
`compare_baseline.py`, which picks `baselines/<environment.baseline_key>.json`
and fails (exit 1) when any gated metric is worse than the baseline beyond
`max(abs, rel * |baseline|)` from `baselines/gate_rules.json`. Accuracy
metrics carry an absolute tolerance of 0.02, timing metrics 15 to 20
percent, counts a small absolute margin; metrics without a rule are printed
for information only. To move the baseline after an accepted change, run
the suite and copy its report over the baseline file for that environment,
then commit it.

`make benchmark_diarization_acceptance RESULT=<report>` checks the Phase 2a
targets in `baselines/phase2a_acceptance.json` and reports each with its
margin, or by how much it was missed. A missed target is never relaxed in
the file; it is reported.

The committed baselines are the official Phase 2 starting point measured on
the unmodified, synchronous pipeline (commit recorded in each file's
`code_revision`): 30 s diarization window inside the caption tick.

### Choosing the window (Phase 2a sweep)

`make benchmark_diarization_window_sweep[_docker]` (`window_sweep.py`)
replays the diarization job loop for every window length and pyannote
segmentation step and prints what each costs and buys. Measured 2026-10-03,
one torch thread, 5 s period, three AMI meetings, first 120 s each, DER on
the labels a viewer sees (first seen):

| window | step | native tick mean / p95 | Linux 4c8g tick mean / p95 | first-seen DER | labels minted per speaker |
|---|---|---|---|---|---|
| 8 s | any | 0.6 s / 0.9 s | 1.4 s / 1.8 s | 0.643 | 1.39 |
| **10 s (default)** | any | **0.58 s / 0.82 s** | **1.33 s / 1.75 s** | **0.627** | **0.94** |
| 12 s | 0.1 | 1.8 s / 2.7 s | 4.0 s / 6.9 s | 0.604 | 0.94 |
| 12 s | 0.25 | 1.2 s / 1.8 s | 2.8 s / 5.7 s | 0.629 | 1.08 |
| 12 s | 0.5 | 1.2 s / 1.6 s | 2.8 s / 3.7 s | 0.653 | 1.17 |
| 15 s | 0.1 | 6.6 s / 11.4 s | 7.3 s / 11.3 s | 0.510 (settled 0.478) | 1.22 |
| 15 s | 0.25 | 1.9 s / 3.1 s | 4.3 s / 7.5 s | 0.561 (settled 0.537) | 0.89 |
| 15 s | 0.5 | 1.24 s / 1.63 s | 2.8 s / 3.7 s | 0.543 | 0.78 |
| 30 s (Phase 2 baseline) | 0.1 | 12.2 s / 15.5 s | 25.6 s / 32.4 s | 0.462 (settled 0.408) | 1.22 |

Two torch threads instead of one (Linux container): 10 s window 1.24 s
mean / 1.75 s p95 against 1.33 s / 1.75 s; 15 s at step 0.5 2.37 s / 3.04 s
against 2.79 s / 3.65 s. The embedding stage barely parallelises, so the
default stays at one thread and the spare cores stay with Whisper.

Readings. A window of 10 s or less is one segmentation window, so the step
is irrelevant and the pass costs one embedding batch: the only setting whose
Linux p95 fits a 5 s period at a 0.3 real-time factor (1.5 s). Every longer
window costs more embedding windows, superlinearly, and the Linux container
is about 2.2 times slower than the laptop at the same thread count. Windows
whose length minus 10 s is not a multiple of the step (12 s at 0.25 or 0.5)
embed a padded chunk and do worse than 10 s. 15 s at step 0.5 (two clean
chunks) is the accuracy-leaning alternative: DER 0.543 for about twice the
pass cost, which would need a 7.5 s period to stay near the RTF budget and
costs label latency. The accuracy cost of leaving the 30 s window is the
price of this phase: first-seen DER 0.462 to 0.627, mostly confusion and
missed speech from clustering on 10 s of context; the reconciler can only
carry identity across the 5 s overlap. Recovering it without the cost is
exactly the embedding-memory work of Phase 2b (`docs/diarization_production_audit.md`,
section 3.2): keep per-label centroid embeddings so a short window can be
clustered against the session's history instead of against itself.
Settings are configuration (`diarization_window_sec`, `segmentation_step`),
so a deployment with cores to spare can take 15 s / 0.5 today.

The full sweep reports are `results/phase2a/window_sweep_native.json` and
`results/phase2a/window_sweep_linux.json` (gitignored; regenerate with the
make targets above).

### Phase 2a results

Measured 2026-10-03 with `make benchmark_diarization_gate_docker` (the
reference container: upstream's CPU image, 4 CPUs, 8 GB nominal, Docker
Desktop on an Apple M4) at this branch, 180 s of ES2004a at real time,
0.5 s frames, reference config (10 s window, 5 s period, one torch thread,
nice 10). The full report is committed as
`baselines/phase2a-linux-cpu-4c8g.json` (a reference, not the gate
baseline: the gate still compares against the Phase 2 starting point, so
that the accuracy cost below stays visible until Phase 2b pays it back).

| metric | diarization off (upstream as shipped) | diarization on |
|---|---|---|
| caption p50, chunk-id in-progress (s) | 5.04 | **4.69** |
| caption p95, chunk-id in-progress (s) | 28.2 | **28.2** |
| caption final p50, chunk-id (s) | 33.6 | 31.4 |
| words first shown p50 / p95 (s) | 11.5 / 28.5 | 10.3 / 28.6 |
| caption periods dropped (of 36) | 9 | **6** |
| audio dropped, buffer full (s) | 0 | 0 |
| transcript messages | 30 | 33 |
| caption worker CPU (cores) / peak RSS (MB) | 1.04 / 931 | 0.98 / 1018 |
| diarization worker CPU (cores) / peak RSS (MB) | - | **0.25 / 881** |
| diarization RTF (compute s per audio s) / pass cost (s) | - | **0.277 / 1.38** |
| diarization passes / dropped periods / audio skipped (s) | - | 36 / 0 / 0 |
| diarization lag p95 (ms) | - | 1775 |
| label after text shown p50 / p95 (s) | - | **0 / 0** |
| finalized words labelled | - | 92% |
| label changes before final / after sent | - | 0 / 0 |
| service RSS peak (MB), two workers | 1226 | 2197 |

Against the Phase 2 baseline in the same container (diarization inside the
caption tick, 30 s window): caption p50 with diarization on went from
39.4 s to 4.7 s, dropped periods from 21 to 6, finalized words from 75 to
262 in the same 180 s, and the pass cost from 25.6 s mean to 1.38 s.

Acceptance (`make benchmark_diarization_acceptance`), container run:

| target | result |
|---|---|
| caption latency (chunk-id) with diarization on within 10% of off | **met**: p50 4.69 s against 5.04 s off, p95 28.2 s against 28.2 s |
| zero caption periods dropped because of diarization | **met**: 6 dropped with diarization on, 9 off (both upstream's own drops under the 4-CPU quota) |
| per session at most 1 core, 1 GB extra RAM, RTF at most 0.3 | **met**: 0.25 cores, 881 MB, RTF 0.277 (margin 0.023; p95 pass 1.69 s against a 5 s period) |
| label at most 2 s p50 / 4 s p95 after the caption text appears | **met**: 0 / 0 s. Captions appear 5 s or more after the audio, labels are ready about 1.4 s after it, so every label was already known when its text arrived (no `speakers_update` was needed in this run); the late path is exercised by the unit tests and by a stalled diarization worker |
| labels change only before finalization, at most 5 percent corrected | **met**: 0 corrections before final, 0 changes after sending |
| streaming settled DER no worse than the Phase 2 baseline | **missed by +0.198**: 0.627 against 0.408 (+0.02 allowed). First-seen DER 0.627 against 0.462 (+0.165); confusion 0.31, missed 0.21, false alarm 0.10 |

What the missed target would take. The sweep above puts the whole gap on
window length: 15 s at step 0.5 recovers 0.08 of it for twice the pass cost
(RTF 0.56 at a 5 s period, or about 0.37 at a 7.5 s period with label
latency rising by up to 2.5 s), and nothing at or under the RTF budget
beats 10 s. Recovering the rest at the 10 s cost is the Phase 2b embedding
memory in the reconciler (`docs/diarization_production_audit.md`, section
3.2): with per-label centroid embeddings a 10 s window is clustered against
the session's history instead of its own 10 s, which is exactly the context
the 30 s window was buying. The audit's own synthetic cases
(`speaker_reconciler_regressions_test.py`, six `xfail` tests) are the
regression suite for that work.

Gate note. `compare_baseline.py` against the Phase 2 starting point fails
on the four accuracy metrics above (first-seen and settled DER and JER)
and on one metric of the **diarization-off** run,
`caption.off.chunk_id.in_progress.p95` (28.2 s against 22.7 s in the
baseline run; p50 improved 6.0 to 5.0 s and dropped periods 11 to 9). The
off run is upstream's code byte for byte on one worker, so that is
run-to-run variance of the VM's single slowest tick, not fork code; the
on run shows the same p95 and the two runs are 30 minutes apart on the
same host.

Concurrency (`make benchmark_diarization_concurrency_docker
SESSIONS="2 3 4"`, same container, same clip streamed over N sessions at
once, diarization on; report committed as
`baselines/phase2a-concurrency-linux-cpu-4c8g.json`). The single-session
runs in this round were noisier than the gate run above (off p50 7.4 s, 13
dropped periods; on p50 8.1 s, 17 dropped), which is the VM's variance, not
a change in code: the two rounds are 40 minutes apart.

| sessions | caption p50 / p95, chunk-id in-progress (s) | caption periods dropped (36 scheduled per session) | caption worker cores | diarization worker cores / RTF | diarization dropped periods / audio skipped (s) | label after text p50 (s) | finalized words |
|---|---|---|---|---|---|---|---|
| 1 (this round) | 8.1 / 41.6 | 17 | 1.2 | 0.25 / 0.28 | 0 / 0 | 0 | 235 |
| 2 | 21.4 / 54.1 | 37 | 1.58 | 0.48 / 0.27 | 0 / 0 | 0 | 156 |
| 3 | 92.4 / 160.8 | 73 | 1.62 | 0.67 / 0.25 | 4 / 0.5 | - | 2 |
| 4 | 91.0 / 162.9 | 103 | 1.63 | 0.80 / 0.28 | 27 / 61 | - | 0 |

**The 4-CPU container sustains one diarized session**, and the limit is
Whisper, not diarization: at two sessions the caption worker (one process,
one job at a time) drops a period per session per period and caption
latency triples, exactly as upstream's own CPU sweeps in
`cpu-findings.md` and the dropped-period counter's design note describe,
while the diarization worker is at half a core with no skipped audio. The
diarization worker only starts to fall behind at four sessions (0.8 cores,
27 of its periods dropped, 61 s of audio skipped), and the design holds
there: the skips show up as `Speaker ?` captions and on the
`diarization_uncovered_seconds` counter, never as caption latency, which
is the same at three and four sessions. Per diarized session the
diarization worker adds about 0.25 cores and shares one 0.9 GB model
process, so on hardware where Whisper serves N sessions the diarization
worker serves the same N up to about four before it needs a second
worker (`worker_ids: [1, 2]`); on a 4-CPU box the question is moot until
upstream's caption path serves more than one session (issue drafts 1 to 3
in `docs/upstream_issue_drafts.md`).


Native run (`make benchmark_diarization_gate`, Apple M4, 16 GB, taken
right after the container runs with the Docker VM still resident: the
hygiene check recorded 7.2 GB of swap in use and 2.8 GB free, so timings
carry paging outliers; informational, the container decides). Report
committed as `baselines/phase2a-native-darwin-arm64.json`.

| metric | diarization off (upstream as shipped) | diarization on |
|---|---|---|
| caption p50 / p95, chunk-id in-progress (s) | 5.04 / 39.4 | **3.19 / 32.0** |
| caption final p50, chunk-id (s) | 35.0 | 14.7 |
| caption periods dropped (of 36) | 13 | **5** |
| audio dropped, buffer full (s) | 15 | 0 |
| diarization worker CPU (cores) / peak RSS (MB) | - | 0.12 / 1376 |
| diarization RTF / pass cost (s) | - | 0.134 / 0.67 |
| diarization lag p95 (ms) | - | 1070 |
| label after text shown p50 / p95 (s) | - | 0 / 0 |
| finalized words labelled | - | 92% |
| label changes before final / after sent | - | 0 / 0 |

Every acceptance target is met natively except the two the container
already shows differently: settled DER (+0.198, the window) and the
diarization worker's peak RSS, 1376 MB against the 1 GB target (881 MB in
the container; macOS reports the Accelerate-backed torch process larger,
and the native number is informational). The native gate also reports the
**diarization-off** run far worse than its own September baseline (p50
5.0 s against 3.0 s, 13 dropped periods against 5, 15 s of audio dropped
against 0): that run is upstream's code on a laptop that was paging, the
same laptop effect the audit's section 1.4 describes, and the on run taken
minutes later on the same machine came out better than it on every
caption metric.


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
replay runs at compute speed; with the Phase 2a tick an hour of audio takes
about ten minutes natively.

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

Diarization costs one pass per period on its own worker. On the 4-CPU
reference container a 10 s window costs about 1.3 s per 5 s period (RTF
0.27) at one torch thread and holds about 0.6 GB; see "Phase 2a results"
for the measured per-session cost and how many sessions the container
sustains. The caption worker is never slowed by it: if the diarization
worker falls behind it skips audio (unlabelled, counted) rather than
queueing. Measure real hardware with `benchmarks/diarization/` before
enabling in production.

## Fork testing checklist

Everything the fork adds can be verified from a clean clone of
`JackyJiang08/ScribeAR-Realtime-Diarization` on `feature/speaker-diarization`.

1. **Unit tests, no models needed.** Python (3.12 + uv):
   `make install_dev_cpu`, `make format`, `make lint` (must score 10/10),
   `make test_unit`. TypeScript (Node 20+): `npm ci`, `npm run build`,
   `npm run lint`, `npm run test:unit`, which includes the speaker-run
   grouping, reducer, label-update, label-slot and WCAG color-contrast
   tests. Two `worker_process_manager` timing tests are known to be flaky
   on loaded laptops and are unrelated to diarization.
2. **End to end with real audio.** Follow "Testing" above: 16 kHz mono WAV
   with at least two speakers, token exported, `diarization_detector: true`,
   the pyannote context on worker 1, `make dev`, then stream the file with
   the manual client. Finalized lines carry inline `[spk_N]` markers; the
   same voice must keep the same label for the whole session, which is the
   `SpeakerReconciler` working.
3. **Evaluation suite.** `make benchmark_diarization_gate_docker` (see
   "Evaluation harness" above) runs the standard suite in the reference
   container and compares it with the committed baseline; then
   `make benchmark_diarization_acceptance RESULT=<report>`.
4. **Full-stack UI check.** Run the Docker Compose stack in `deployment/`,
   join a session from the client webapp and speak with two people: captions
   appear with a `Speaker ?` slot that fills in with `Speaker 1:` /
   `Speaker 2:` within a few seconds, in distinct colors that stay readable
   (WCAG AA, contrast at least 4.5:1) on any background theme, and no text
   moves when a label arrives.

## What the fork adds, by layer

| Layer | Change |
|---|---|
| `transcription_service` | `PyannoteDiarizationContext`: optional worker-pool context running `pyannote/speaker-diarization-community-1` (CPU by default, CUDA-ready) with its own thread cap, nice and segmentation step |
| `transcription_service` | `DiarizationJob`: a second worker-pool job per session on the diarization worker, newest-window passes, skip-based back-pressure, lag and uncovered-audio counters |
| `transcription_service` | `SpeakerReconciler`: maps per-pass labels to session-wide labels by overlap voting, so one voice keeps one label across passes |
| `transcription_service` | `SpeakerLabelAttacher`: joins captions and labels in the session without ever holding a caption; frozen label timeline, late `speakers_update`s, timeout |
| `transcription_service` | Diarization metrics on `/metrics/status`, folded apart from the caption series |
| Wire format | Optional `speakers` and `sequence_id` on transcript sequences and the `speakers_update` message, end to end through the WebSocket messages and the shared TypeScript schemas; all null or absent when diarization is off |
| node-server | Relays `speakers_update` as `speakersUpdate` on its own bus channel |
| Client UI | Content store `applySpeakersUpdate` (fills nulls only); display `SpeakerLabelSlot` placeholder that fills in place; `Speaker N:` labels colored from a colorblind-aware palette auto-adjusted to WCAG AA contrast against the configured background |
| Monitoring | Sidecar `scribear_diarization_*` series, `diarizationBehindRule`, two Grafana panels |
| Tooling | Manual end-to-end client, evaluation harness (replay benchmark, caption-latency probe with label latency and per-worker cost, window sweep, concurrency, Linux CPU reference container, regression gate, acceptance targets, hard cases, soak), this document |

Planned follow-ups beyond diarization: resumable lecture summarization
(prompt-injection resistant) and RNNoise-based denoising.

This fork is an independent feature-development copy built on ScribeAR by the
ScribeAR team at the University of Illinois Urbana-Champaign; upstream retains
all rights to the original code.

## Known limitations

- A 10 s diarization window loses accuracy against the 30 s one it
  replaced (first-seen DER 0.627 against 0.462 on the AMI set): clustering
  has 10 s of context and the reconciler carries identity across only the
  5 s overlap. Labels can still drift after long silences or when a voice
  is absent from the overlap; the reconciler then mints a fresh label for
  what may be the same voice. Fixing this properly requires
  speaker-embedding memory, which is Phase 2b.
- Words already finalized keep the label they were finalized with; later,
  better speaker evidence does not retroactively update them. That is by
  design: a label shown on a finalized caption never changes.
- After the caption worker drops audio (its buffer was full for longer
  than it holds), the drop is placed at the newest Whisper time the session
  had seen, so words between the drop and the next result can be a few
  seconds off against the diarization clock.
- Accuracy degrades with heavily overlapping speech and far-field audio.
- GPU deployments can set `"device": "cuda"` in the context config; a
  streaming-native alternative (NVIDIA Streaming Sortformer) can be added as
  another context implementation without touching the provider wiring.
