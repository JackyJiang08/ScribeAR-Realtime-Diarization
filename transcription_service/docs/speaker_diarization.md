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
   window" for why). On a window of one segmentation chunk (10 s or less)
   the pyannote context reports the segmentation model's own speaker
   tracks, each with its centroid embedding, rather than the pipeline's
   clustered output (which collapses such a window to one speaker; see
   "Phase 2b results"), and runs the embedding network once per window,
   pooling it once per active track, instead of once per speaker slot
   (same embeddings at a third of the cost; see "Phase 2 wrap-up
   results"). `SpeakerReconciler` then maps the pass's tracks
   onto stable session labels by **speaker-embedding memory**: it keeps one
   running centroid per session speaker, in memory only, and attaches each
   track to the speaker whose centroid it resembles the most, wherever and
   whenever that speaker last spoke. A voice keeps its label when the
   window slides, after silence and after a long gap. A new label is minted
   only for a track that is clearly far from every known speaker and has
   enough speech behind it; shorter unknown voices are pooled until they
   have been heard enough, and backchannels attach to the best match or
   stay unlabelled. Because consecutive windows overlap, the audio a new
   label was minted on is diarized again by the next pass: when that pass
   labels most of it as an existing speaker, the new label was a split of
   that speaker and is **folded** into it before its captions settle (its
   centroid is parked, so a voice that does come back takes its label
   back). The pass reports the labelled segments, the score each
   label was attached with, the window it covered and how old the audio
   was when the labels were ready (the **diarization lag**).

The session (main process) joins the two with `SpeakerLabelAttacher`:

- Every caption result is forwarded immediately. Words whose audio
  diarization has already covered carry their label in the `speakers`
  array; the rest carry `null`. A finalized sequence is sent with a
  session-unique `sequence_id`.
- Every diarization result extends a label timeline up to
  `window_end - diarization_edge_margin_sec` (the last half second of a
  window is left for the next pass, which sees it with context). Where two
  passes cover the same audio (consecutive windows overlap by window minus
  period), the earlier label stands unless the later pass attached its
  label with clearly better evidence (`diarization_revision_margin` more
  score) or the earlier pass found no speech there. Such a **revision**
  reaches only words that are not finalized yet: the in-progress tail,
  which the webapps re-render every tick anyway, and finalized words still
  waiting for their first label. A label sent on a finalized sequence is
  never sent differently afterwards. Finalized words that were still
  `null` are sent later as a `speakers_update` naming the `sequence_id`; a
  client that ignores the message still has every caption.
- A decided word that no speaker segment overlaps (Whisper heard a word
  where pyannote found no speech: a quiet word in a pause inside a turn)
  takes the speaker of the nearest segment within
  `diarization_attach_gap_sec` (1 s), so it is not left unattributed.
- A finalized sequence whose words were all decided at emission but some
  got no label (no speaker segment overlapped them) is settled at once with
  a `speakers_update`, so the client can show an explicit unattributed
  state instead of waiting. Audio no pass covered (the diarization job fell
  behind and skipped ahead) is decided as "no speaker" the moment the next
  pass reports its window, and a finalized sequence that has waited longer than
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

**Reconnects.** node-server reconnects to the service after any blip and
sends the same `session_uid` again. When a session ends, the provider keeps
its speaker memory (the reconciler's labels and centroid embeddings, and
the sequence-id counter) in memory for `diarization_reconnect_grace_sec`
(60 s); a new session with the same uid inside that grace starts its
diarization job from it, so the same people keep the same labels and no
sequence id is ever reused. The memory is held only in the service
process, is handed over once, and is dropped for good when the grace
expires; nothing is ever written to disk. Voiceprints are biometric data,
and this is the whole of their lifetime.

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

   Speaker identity settings (Phase 2b; the defaults were tuned on the AMI
   benchmark set, see "Phase 2b results", and rarely need changing):

   | field | default | meaning |
   |---|---|---|
   | `diarization_match_threshold` | 0.4 | cosine similarity (plus overlap bonus) at or above which a track attaches to the best-matching known speaker |
   | `diarization_new_speaker_threshold` | 0.3 | a track mints a new speaker only when its best score against every known speaker is below this |
   | `diarization_attach_threshold` | 0.2 | a track too short to mint attaches to the best speaker at or above this; below it the track stays unlabelled and is pooled |
   | `diarization_min_mint_sec` | 2.5 | speech a voice needs, in one pass or accumulated over the passes that heard it, before it can mint a label |
   | `diarization_max_session_speakers` | 32 | bound on labels (and centroids) per session |
   | `diarization_merge_threshold` | 1.0 (off) | two speakers whose centroids reach this similarity are merged; re-tested in "Phase 2 wrap-up results" (never removes a fragment, merges real people below 0.5) |
   | `diarization_merge_max_age_sec` | 20 | with merging on, only a speaker minted within this many seconds can be merged away (0 = any age), so two established speakers are never joined |
   | `diarization_fragment_fold_sec` | 7.5 | a label minted within this many seconds whose audio the next pass re-labels as another speaker is folded into that speaker before its captions settle; 7.5 s is one and a half periods at the default 5 s period (exactly the next pass), scale it with `diarization_period_ms`; 0 disables. Measured in "Phase 2 wrap-up results" |
   | `diarization_fragment_fold_fraction` | 0.8 | fraction of the fragment's audio inside the next window that other speakers must cover for the fold |
   | `diarization_overlap_bonus` | 0.15 | added to the score in proportion to the time overlap with the previous pass's labels (consecutive windows share audio); also carries identity when a pass has no embeddings |
   | `diarization_sustained_split_sec` | 0 (off) | a voice that keeps scoring between the new-speaker and the match threshold against its best speaker mints its own label after this many seconds of such evidence; measured in "Phase 2c results" (adds labels, never finds a second person on the benchmark set) |
   | `diarization_recluster_period_sec` | 0 (off) | every this many seconds of session time the session's track history is re-clustered with the PLDA/VBx clustering shipped with the model, merging and splitting speakers to follow it; sent labels never change. Measured in "Phase 2c results" |
   | `diarization_attach_gap_sec` | 1.0 | a word with no speech segment under it takes the nearest speaker within this gap; 0 disables |
   | `diarization_revision_margin` | 0.1 | a later pass replaces an earlier label on shared audio only when it is at least this much more confident |
   | `diarization_reconnect_grace_sec` | 60 | how long a closed session's speaker memory is kept for a reconnect; 0 disables |

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
| `local_speakers` | `true` | on a one-chunk window (10 s or less) report the segmentation model's speaker tracks with their embeddings instead of the pipeline's clustered output, which collapses such a window to one speaker. Longer windows always use the pipeline's clustering. |
| `overlap_aware` | `false` | report pyannote's overlap-aware turns instead of the exclusive ones; words are attributed to the speaker overlapping them the most either way. Measured in "Phase 2b results". |
| `clustering_threshold` | `null` (model 0.6) | override of the pipeline's VBx clustering threshold; only matters for windows longer than one chunk. |
| `shared_embeddings` | `true` | with `local_speakers`, run the embedding network's frame stage once per window and pool it per active speaker track instead of pyannote's one full pass per speaker slot (three per window). Same embeddings, about a third of the pass cost; `false` runs the pipeline unchanged. Measured in "Phase 2 wrap-up results". |

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
  never changed, by the service and by the clients alike. Corrections
  happen before that point: the service may revise a label while the
  word's sequence is still in progress, never after the finalized
  sequence carried it.

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
without a label empties the slot too (`data-speaker-slot="unattributed"`):
a `Speaker ?` never stays on screen once the provider has said nothing more
will come, and the words read as plain caption text. Speaker changes inside
one sequence are labelled inline, as before. The slot is
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

1. **Replay benchmark** (`benchmark_baseline.py`) over the evaluation
   set (`SET=standard`, the default since Phase 2c: the 16 AMI test-set
   meetings plus the VoxConverse subset, single distant microphone or
   in-the-wild audio, first 10 min; `SET=dev`: ES2004a, IS1009a, TS3003a,
   the Phase 2 meetings; `eval_sets.py`): one offline pass per file of
   `OFFLINE_SET` (default `dev`, since a whole-file pass costs minutes on
   CPU; `make benchmark_diarization_offline_counts` covers the standard
   set once) and a streaming replay of the diarization job loop (the
   reference config's period and window, `SpeakerReconciler`) over the
   first `STREAM_SEC` seconds (default 0 = the full 10 min since Phase 2c;
   the Phase 2a/2b baselines used 120).
2. **Caption latency** (`caption_latency.py`): starts the real service with
   the reference config, streams ES2004a at real time in 0.5 s SAFP frames
   for `CAPTION_SEC` seconds (default 180), diarization **on** and **off**.
3. **Classroom case** (`classroom_score.py` over a replay of
   `data/classroom`, see "Phase 2c results"): whether each short question
   gets a label other than the instructor's.
4. Optionally **concurrency** (`SESSIONS="2 3 4"` with
   `make benchmark_diarization_concurrency_docker`): the same stream over N
   concurrent sessions with diarization on.

### Metrics

Per file and aggregated, in `report.replay`:

- `offline` / `streaming_first_seen` / `streaming_settled`: **DER** with
  its missed / false-alarm / confusion breakdown (pyannote convention: no
  collar, overlap scored; a 0.25 s collar variant is included), **JER**,
  and the **speaker count error** (hypothesis labels minus reference
  speakers; since Phase 2c a reference speaker counts only with at least
  `--min-speaker-sec` = 5 s of speech in the streamed part, the raw count
  is kept beside it, and the report lists true and predicted counts per
  file in `speaker_counts` together with the signed mean error, the
  fraction of files within one and exact, and how many were under- and
  over-counted). *First seen* is the label the newest pass gives in-progress
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
  created divided by the people actually speaking in the streamed part
  (at least 5 s). Two-sided since Phase 2c: the acceptance band is 0.8 to
  1.2, because "at most 1.2" could not catch under-counting. Since the
  Phase 2 wrap-up every count metric also exists on the labels present on
  **settled captions** (`labels_settled_per_reference_speaker`,
  `speaker_count_settled_within_1_fraction`,
  `speaker_count_settled_exact_fraction`): a label the fragment fold
  removes before its audio settles is minted but never shown, so these
  are what a viewer sees and what the band and the gate
  (`labels_settled_per_speaker_deviation`, distance from 1) use. The
  `*_all` variants count every reference speaker however briefly they
  spoke, beside the 5 s rule.
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
margin, or by how much it was missed; `TARGETS=` points it at the Phase 2b
(`phase2b_acceptance.json`), Phase 2c (`phase2c_acceptance.json`, with
the two-sided labels-per-speaker band and the classroom target) and
Phase 2 wrap-up (`phase2_wrapup_acceptance.json`: RTF at most 0.25, at
most 1.5 labels per person on the classroom's settled captions, the band
and the count on settled captions) files. A missed target is never
relaxed in the file; it is reported.

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
| label at most 2 s p50 / 4 s p95 after the caption text appears | **met**: 0 / 0 s, measured over the 92 percent of finalized words that received a label (242 of 262): captions appear 5 s or more after the audio, labels are ready about 1.4 s after it, so every label was already known when its text arrived and no `speakers_update` was needed in this run; the late path is exercised by the unit tests and by a stalled diarization worker. The other 8 percent are words diarization covered but no speaker segment overlapped (pyannote found no speech there: hallucinated or far-field words in pauses, the same words that go unlabelled offline); they keep the `Speaker ?` placeholder, settled at once so the client knows nothing more will come |
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


### Phase 2b results

Speaker identity quality, measured 2026-10-03. The replay numbers come
from `tune_reconciler.py` over the three AMI meetings; every row below
runs the production `SpeakerReconciler` and `SpeakerLabelAttacher` over
the same cached pyannote passes (10 s window, 5 s period), so differences
are the reconciler's alone. "Full" scores the whole 10 min of each file,
"120 s" the gate horizon (the first two minutes, where every meeting is
still introductions and most speakers have not spoken yet).

**What the window hides.** On a window of one segmentation chunk the
pipeline's VBx clustering collapsed every window to one speaker: 0 of 360
passes carried a second label on the exclusive output, although the
segmentation model separated two voices inside 110 of them. Two chunks
(15 s at step 0.5) did no better (0 of 360). So until Phase 2b every turn
change inside a window became confusion, and this, not the window length,
was the floor of the Phase 2a numbers. The context now reports the
segmentation's own speaker tracks with their embeddings (`local_speakers`)
and leaves the clustering to the reconciler's session memory:

| pass output | settled DER full / 120 s | confusion full / 120 s | labels per speaker | speaker count within one |
|---|---|---|---|---|
| pipeline clustering, 10 s (exclusive) | 0.291 / 0.357 | 0.074 / 0.073 | 1.17 / 0.67 | 1 of 3 / 2 of 3 |
| pipeline clustering, 15 s at step 0.5 | 0.336 / 0.411 | 0.120 / 0.105 | 0.92 / 0.75 | 2 of 3 / 2 of 3 |
| **segmentation tracks, 10 s (default)** | **0.279 / 0.313** | **0.061 / 0.021** | 0.83 / 0.64 | 2 of 3 / 2 of 3 |

(Each row at its own best thresholds; the 15 s row costs 2.5 times the
pass and already failed the Phase 2a RTF budget, so the window stays at
10 s.)

**Threshold tradeoff** (segmentation tracks, full 10 min; the gate
horizon ranks the same way). The match threshold hardly matters between
0.4 and 0.6; what trades off is how readily a new voice mints a label:

| new-speaker threshold | minting minimum | overlap bonus | settled DER | confusion | labels per speaker | count within one |
|---|---|---|---|---|---|---|
| 0.3 | 2.5 s | 0 | 0.271 | 0.053 | 1.17 | 1 of 3 |
| **0.3** | **2.5 s** | **0.15** | **0.279** | **0.061** | **0.83** | **2 of 3** |
| 0.3 | 4 s | 0 | 0.280 | 0.062 | 0.92 | 1 of 3 |
| 0.4 | 4 s | 0 | 0.281 | 0.061 | 1.67 | 1 of 3 |
| 0.5 | 6 s | 0 | 0.325 (120 s) | 0.033 | 0.83 | 2 of 3 |

The default (second row) gives up 0.008 DER against the first for a
third fewer labels and the better speaker count. Per meeting it mints 5
labels for the 4 speakers of ES2004a (a noisy far-field room), 4 for 4 on
IS1009a, and 1 for 4 on TS3003a, whose voices the embedding model keeps
within similarity 0.5 of each other: a lower threshold splits ES2004a
further before it separates TS3003a, which is why the speaker-count target
(within one on 80 percent of meetings) is missed at 2 of 3 whichever way
the threshold moves. Merging converged speakers never fired (the spurious
labels are not near any centroid: maximum pairwise similarity 0.53), so it
ships disabled. Revising a label on shared audio when the later pass is
0.1 more confident is worth 0.003 to 0.015 DER for 0.7 to 1.7 percent of
audio revised; a smaller margin revises more and gains nothing.

**Overlap.** Scoring pyannote's overlap-aware turns instead of the
exclusive ones changed settled DER by less than 0.002 at every threshold
(0.290 against 0.291 full, 0.359 against 0.357 at 120 s), because the
second label the pipeline adds in overlap carries no embedding and the
words there are attributed to the dominant speaker either way. Overlapping
speech therefore stays shown as the dominant speaker's words; the
`overlap_aware` switch is kept for measurement. Missed speech (0.16 of
reference speech) is the largest remaining term and is the segmentation
model's: the community-1 segmentation is a powerset model with no
detection threshold to lower, and `min_duration_off` is already 0.

**Gate runs.** Measured with `make benchmark_diarization_gate_docker` and
`make benchmark_diarization_gate` before (commit d78abef, Phase 2a code)
and after (this branch), same clips, same configs, 120 s replay per file
and 180 s of ES2004a through the real service. The replay is
deterministic and gave the same accuracy numbers in both environments;
the caption and resource columns are the container's (the native laptop
run carried a paging warning both times and is informational).

| metric | Phase 2 baseline (30 s window in the tick) | before 2b (Phase 2a) | after 2b |
|---|---|---|---|
| streaming first-seen DER | 0.462 | 0.627 | **0.340** |
| streaming settled DER | 0.408 | 0.627 | **0.313** |
| settled confusion / missed / false alarm | 0.095 / 0.208 / 0.106 | 0.312 / 0.213 / 0.102 | **0.021** / 0.152 / 0.139 |
| settled JER | 0.620 | 0.818 | **0.546** |
| labels minted per reference speaker | 1.22 | 0.94 | 0.64 |
| speaker count within one | - | 1 of 3 | 2 of 3 |
| label flip rate after first shown | 0.19 | 0.0 | 0.0 |
| replay tick p95 (container) | 32.4 s | 2.26 s | 1.67 s |
| caption p50 / p95, chunk-id, diarization on (container) | 39.4 / 72.3 s | 4.55 / 22.2 s | 6.23 / 23.0 s |
| caption p50 / p95, diarization off (container) | 6.0 / 22.7 s | 5.53 / 22.1 s | 6.74 / 59.6 s |
| caption periods dropped, on / off (container) | 21 / 11 | 10 / 8 | 11 / 17 |
| diarization worker cores / RSS / RTF (container) | - | 0.25 / 866 MB / 0.277 | 0.27 / 847 MB / 0.299 |
| finalized words labelled | 77% | 92% | **99.6%** (246 of 247) |
| label after text p50 / p95 | - | 0 / 0 s | 0 / 0 s |
| corrections before final / changes after sent | - | 0 / 0 | 0 / 0 |

Per meeting (120 s, both environments): ES2004a settled DER 0.468 with
2 labels for 3 speakers, IS1009a 0.259 with 4 for 4, TS3003a 0.241 with
1 for 4 (confusion 0.0: nobody is mislabelled, the other three voices
count as one). Natively the probe labelled 100 percent of finalized words
(259 of 259) with captions at 4.83 s p50 against 5.65 s before.

**Hard cases** (`make benchmark_diarization_hardcases`, now at the
production window; each case streamed in full, no earlier run at 10 s to
compare against):

| case | reference speakers | labels minted | settled DER | confusion | missed | false alarm |
|---|---|---|---|---|---|---|
| four_speakers | 4 | 4 | 0.406 | 0.207 | 0.182 | 0.018 |
| short_turns | 4 | 4 | 0.392 | 0.200 | 0.081 | 0.111 |
| noise_clean | 4 | 4 | 0.392 | 0.200 | 0.081 | 0.111 |
| noise_pink_snr5 | 4 | 2 | 0.569 | 0.352 | 0.089 | 0.128 |
| overlap | 4 | 3 | 0.402 | 0.119 | 0.255 | 0.028 |
| return_after_gap | 4 | 3 | 0.333 | 0.062 | 0.140 | 0.131 |

Pink noise at 5 dB SNR is the one case that breaks identity: the
embeddings of two of the four voices fall within the match threshold of
each other and the session ends with two labels. The returning speaker
(FEE016, silent for 253 s) comes back under the label it had, which is the
case the old reconciler failed by construction.

A first container "after" run was discarded and repeated: one Whisper
execution on the caption worker took 95 s, the service dropped 69.5 s of
audio, and only 103 words were finalized, which left 8 percent of them
unlabelled and tripled caption latency with diarization on; the
diarization worker's own cost in that run was identical to the before run
(54 against 50 CPU seconds), the worker that stalled runs upstream's
unchanged job, and the repeat on a quieter host showed none of it. It is
the VM outlier the audit's section 1.4 describes, reported here because
it is also what the 15 s label timeout is for: those captions settled
unlabelled rather than waiting. Reports:
`baselines/phase2b-linux-cpu-4c8g.json` (the repeat),
`baselines/phase2b-native-darwin-arm64.json`, and the discarded run in
`results/phase2b/after_linux.json` (gitignored).

Acceptance (`make benchmark_diarization_acceptance`, container repeat):
every Phase 2a target is met (caption p50 6.23 s on against 6.74 s off,
p95 23.0 against 59.6 s, 11 dropped periods against 17 off, 0.27 cores,
847 MB, RTF 0.299 against the 0.3 limit, labels 0 s after text, no label
changed after sending, no correction before final). Phase 2b
(`TARGETS=benchmarks/diarization/baselines/phase2b_acceptance.json`):

| target | result |
|---|---|
| must: settled DER better than the pre-2a 0.408 | **met**: 0.313 (margin 0.094) |
| must: at least 97% of final words labelled | **met**: 99.6% |
| must: the six expected-fail reconciler tests pass, markers removed | **met** (`speaker_reconciler_regressions_test.py`) |
| target: settled DER at most 0.32 | **met**: 0.313 (margin 0.007; on the full 10 min of each meeting 0.279) |
| target: speaker confusion at most 0.08 | **met**: 0.021 |
| target: at most 1.2 labels minted per real speaker | **met**: 0.64 |
| target: speaker count within one on at least 80% of meetings | **missed by 13 points**: 2 of 3 meetings (67%). TS3003a ends with one label for four voices whose embeddings stay within similarity 0.5 of each other; lowering the new-speaker threshold enough to split them raises ES2004a to 8 or more labels (see the tradeoff table), so the threshold stays where DER, confusion and labels per speaker are best |
| at most 5% of segments corrected before finalization | **met**: 0 in the probe; 0.7% of audio revised in the replay |

RTF sits at 0.299 against the 0.3 ceiling (0.277 before): the local
speaker tracks add a few milliseconds of numpy per pass and the pass
costs 1.49 s mean against 1.39 s. A deployment that needs headroom can
raise `diarization_period_ms` to 6000 (RTF about 0.25, labels 1 s later).

The gate against the Phase 2 starting point still fails, on five metrics
that all belong to the **diarization-off** run: p95 59.6 s against
22.7 s, 17 dropped periods against 11, 30 s of audio dropped against
4.3 s, and the word-method first-shown p50 and p95. That run is upstream's
code byte for byte on one worker, so this is the VM's single slowest tick
(the audit's section 1.4), not fork code; every accuracy metric passes
the old gate with room to spare. With every must target met the baseline
is moved to this run in its own commit, and the Phase 2b numbers become
the gate.

### Phase 2c results

Speaker counting on a broader set, measured 2026-10-03/04. Phase 2b met
every must target but minted 0.64 labels per real speaker on the gate
horizon, and TS3003a ended with one label for its four reference speakers:
a sign that different people were being merged into one label, the
visible error in a classroom. This phase asks whether that is so, on
enough meetings to tell, and fixes what the numbers support.

**Evaluation set.** The standard set is now every meeting of the AMI test
set (pyannote/AMI-diarization-setup `lists/test.meetings.txt`: ES2004a-d,
IS1009a-d, TS3003a-d, EN2002a-d; Array1-01, first 10 minutes, only_words
references) plus a VoxConverse v0.3 test subset (`voxconverse_subset.json`:
for each speaker count from 1 to 8 the longest test file, first 10
minutes; political debates and news panels, CC BY 4.0, read from the
test archive with HTTP range requests so 170 MB are transferred instead
of 4.3 GB). The three Phase 2 meetings remain the `dev` set
(`SET=dev`). `eval_sets.py` defines both; `make benchmark_diarization_data`
prepares everything. Audio is never committed. Every number below is on
the full 10 minutes of each file (the Phase 2a/2b tables scored the first
120 s, where most speakers had not spoken yet).

**Two-sided counting.** Labels minted per real speaker now has a band,
0.8 to 1.2, in `phase2c_acceptance.json`; the gate tracks its distance
from 1 (`replay.labels_per_speaker_deviation`). The report lists the
true and predicted speaker count per meeting (`speaker_counts`), the
signed mean error and how many meetings were under- and over-counted. A
reference speaker counts only with at least 5 s of speech in the scored
part (`--min-speaker-sec`; the raw count is kept beside it), because a
voice with a few seconds in ten minutes cannot earn a label: the minting
minimum alone is 2.5 s of evidence. This matters for TS3003a: in its
first ten minutes MTD009PM speaks for 490 s and the other three for 4.0,
11.6 and 1.6 s, so "one label for four voices" was mostly a reference
artefact, and the one real miss is the 11.6 s speaker.

**Model or pipeline?** Two measurements separate what the embedding model
can tell apart from what the reconciler does with it.
`embedding_separability.py` takes every pure segmentation track of the
cached passes (at least 1 s, 70 percent inside one reference speaker),
scores every pair with cosine similarity and with the PLDA
log-likelihood ratio that ships with community-1, and reports the best
achievable error on each meeting:

| meeting | tracks | cosine EER (threshold) | PLDA EER (threshold) | same-speaker pairs below 0.4 | different-speaker pairs at or above 0.4 | nearest oracle centroids (cosine) |
|---|---|---|---|---|---|---|
| ES2004a | 111 | 0.170 (0.21) | 0.197 (2.6) | 44% | 0.3% | 0.15 |
| IS1009a | 123 | 0.122 (0.27) | 0.132 (0.9) | 25% | 2.1% | 0.03 |
| TS3003a | 120 | one speaker has pure tracks | - | 18% | - | - |

The voices are separable: the real speakers' centroids sit at cosine 0.15
or less from each other, and only 0.3 to 2 percent of different-speaker
track pairs reach the match threshold. What is noisy is the single
far-field track: a quarter to almost half of same-speaker pairs fall
below 0.4. So the risk at the 10 s window is splitting one person into
several labels, not merging two people into one, and the shipped PLDA is
no sharper than cosine on these tracks (its EER is slightly worse on both
meetings), which is why the clustering experiments below do not improve
on the cosine rule. The second measurement, offline pyannote over the
whole 10 minutes of every file, is in the table below (column "offline").

**Classroom scenario.** `prepare_classroom_case.py` builds a synthetic
lecture from the four people of the ES2004 series (same room, same
distant microphone): stretches where only one person speaks, joined with
0.4 s silences; the speaker with the most such speech (FEE013, 486 s) is
the instructor and talks in 45 s blocks, the other three ask three
questions each of 5 to 15 s in a seeded order, and one questioner's
questions (MEE014) carry pink noise at 5 dB SNR. The case is 552 s long,
the instructor has 83 percent of the speech, the questions total 95 s.
`classroom_score.py` reads the settled label timeline and gives every
question the label covering most of it: a question has "its own label"
when that label is not the instructor's (the label covering most of the
lecture time). The case runs as a suite step (`classroom.*` key metrics,
`make benchmark_diarization_classroom`).

**Under-counting fixes, measured.** All replays below run the production
reconciler and attacher over cached pyannote passes (10 s window, 5 s
period), so a configuration costs seconds to score and the pyannote stage
is identical for every row. Scored on the dev set plus the classroom case,
full length:

| configuration | settled DER | confusion | labels per speaker | count within one | signed count error | labels: ES2004a, IS1009a, TS3003a, classroom (reference 4, 4, 2, 4) |
|---|---|---|---|---|---|---|
| Phase 2b defaults | **0.231** | **0.061** | 1.19 | 3 of 4 | +1.00 | 5, 4, 1, 8 |
| (b) sustained split 15 s | 0.231 | 0.061 | 1.50 | 2 of 4 | +2.00 | 6, 5, 2, 9 |
| (b) sustained split 10 s | 0.333 | 0.163 | 1.62 | 1 of 4 | +2.50 | 6, 6, 2, 10 |
| (a) PLDA/VBx re-clustering every 60 s | 0.246 | 0.076 | 1.25 | 3 of 4 | +0.75 | 4, 4, 1, 10 |
| (a) re-clustering every 120 s | 0.260 | 0.090 | 1.19 | 3 of 4 | +0.75 | 4, 4, 1, 9 |
| (c) minting minimum 4 s | 0.239 | 0.068 | 1.00 | 3 of 4 | +0.25 | 4, 3, 1, 7 |
| (c) a voice must be re-found in the next window (`min_mint_passes` 2) | 0.287 | 0.109 | 0.81 | 2 of 4 | -0.50 | 2, 3, 1, 6 |
| (c) new-speaker threshold 0.25 | 0.279 | 0.110 | 0.88 | 2 of 4 | -0.25 | 4, 2, 1, 6 |

- *(b) Sustained split.* A voice that keeps scoring between the
  new-speaker and the match threshold against its best speaker mints its
  own label after N seconds of such evidence (`sustained_split_sec`;
  grey-zone attachments stop moving the centroid so the split stays
  possible). It never finds a second person: on TS3003a's single speaker
  it splits one far-field voice in two at 10 s (confusion 0.008 to
  0.393, the two labels then alternate), and at 15 s it adds one spurious
  label to every file. Shipped disabled.
- *(a) Session-level re-clustering.* Every N seconds the session's track
  history (raw embeddings, up to 600) is re-clustered with the PLDA/VBx
  clustering of community-1 at its shipped hyper-parameters (AHC 0.6, Fa
  0.07, Fb 0.8; tracks weighted by their seconds), speakers whose tracks
  the clustering joins are merged and a speaker whose tracks fall into
  two clusters is split; sent labels never change. On oracle tracks the
  clustering recovers the four speakers of ES2004a and IS1009a cleanly
  (and one cluster for TS3003a, correctly), and in the replay it fixes
  ES2004a (5 labels to 4, DER 0.412 to 0.371). But in the classroom it
  merges the 35 s questioner FEE016 (cosine 0.39 to the instructor) into
  the instructor's 840 s and splits the noisy questioner in two, so
  confusion there rises from 0.060 to 0.148 and the mean gets worse. The
  machinery stays (`diarization_recluster_period_sec`, 0 = off) because it
  is the right tool when a deployment has long sessions of a few
  long-speaking people; it is not the default.
- *(c) Stricter minting.* Fewer labels for more confusion and DER in
  every variant: the extra labels the defaults mint are fragments (in the
  classroom: two labels on the instructor's first seconds before the
  centroid settles, a 0 s and a 2 s label), and making minting stricter
  removes real speakers before it removes fragments. Merging fragments
  (labels with little speech) into their nearest speaker afterwards was
  also tried and cannot help the viewer-centric count, since a label
  already shown stays shown.

**On the standard set** (24 files, full 10 minutes, replay from the
cached passes; the container run below reproduces these accuracy numbers
exactly, the replay being deterministic):

| configuration | settled DER | confusion | labels per speaker | count within one | exact | signed error (under / over) |
|---|---|---|---|---|---|---|
| **Phase 2b defaults (kept)** | 0.267 | 0.103 | **1.17** | **20 of 24 (83%)** | 8 of 24 | +0.42 (4 / 12) |
| (a) re-clustering every 60 s | 0.244 | 0.080 | 1.49 | 16 of 24 | 7 | +1.29 (2 / 15) |
| (a) re-clustering, merges only | 0.270 | 0.106 | 1.29 | 16 of 24 | 10 | +0.67 |
| (b) sustained split 15 s | 0.257 | 0.093 | 1.44 | 13 of 24 | 5 | +1.25 (1 / 18) |
| (a) + (b) | 0.231 | 0.067 | 1.65 | 13 of 24 | 5 | +1.83 (0 / 19) |
| (c) minting minimum 4 s | 0.277 | 0.111 | 0.92 | 20 of 24 | 7 | -0.42 (12 / 5) |
| (c) 4 s + (b) 15 s | 0.265 | 0.100 | 1.19 | 20 of 24 | 9 | +0.46 (3 / 12) |

Re-clustering and the sustained split buy DER and confusion with labels:
(a) + (b) reaches 0.231 / 0.067 but mints 1.65 labels per speaker and
over-counts 19 of 24 meetings, because each split lands a new label on
audio whose earlier label stays on screen. Nothing improves the count
without raising confusion, so the Phase 2b thresholds stay and both
guards ship disabled (`diarization_sustained_split_sec`,
`diarization_recluster_period_sec`, both 0). Per meeting, with the
defaults (reference speakers with at least 5 s of speech; in brackets
with any speech; "offline" is pyannote's own pipeline over the whole
file, run once on the dev meetings and on the six files whose count the
replay got wrong or most inflated, "-" = not run):

| file | reference | predicted | offline | settled DER | confusion |
|---|---|---|---|---|---|
| IS1009a | 4 | 4 | 4 | 0.290 | 0.078 |
| IS1009b | 3 (4) | 4 | - | 0.146 | 0.056 |
| IS1009c | 2 (4) | 2 | - | 0.123 | 0.019 |
| IS1009d | 4 | 5 | 4 (DER 0.124) | 0.303 | 0.194 |
| ES2004a | 4 | 5 | 4 | 0.412 | 0.105 |
| ES2004b | 4 | 6 | 4 (DER 0.176) | 0.200 | 0.030 |
| ES2004c | 4 | 5 | - | 0.441 | 0.297 |
| ES2004d | 4 | 4 | - | 0.350 | 0.159 |
| TS3003a | 2 (4) | 1 | 1 | 0.158 | 0.008 |
| TS3003b | 3 (4) | 4 | - | 0.130 | 0.013 |
| TS3003c | 3 | 3 | - | 0.398 | 0.264 |
| TS3003d | 4 | 4 | - | 0.379 | 0.206 |
| EN2002a | 4 | 5 | - | 0.393 | 0.132 |
| EN2002b | 4 | 7 | 4 (DER 0.422) | 0.560 | 0.185 |
| EN2002c | 3 | 4 | - | 0.252 | 0.035 |
| EN2002d | 4 | 3 | 3 (DER 0.401) | 0.411 | 0.079 |
| VoxConverse bgvvt | 2 | 2 | - | 0.114 | 0.001 |
| VoxConverse epygx | 5 | 5 | - | 0.196 | 0.079 |
| VoxConverse gtjow | 2 | 2 | - | 0.065 | 0.002 |
| VoxConverse hhepf | 6 | 2 | 6 (DER 0.077) | 0.422 | 0.360 |
| VoxConverse iacod | 3 | 2 | 3 (DER 0.071) | 0.155 | 0.087 |
| VoxConverse jwggf | 3 (5) | 6 | - | 0.114 | 0.017 |
| VoxConverse uicid | 1 | 2 | - | 0.114 | 0.005 |
| VoxConverse ylgug | 2 (3) | 3 | - | 0.064 | 0.007 |

The four under-counted files are TS3003a (the 11.6 s speaker), EN2002d
(3 for 4), iacod (2 for 3) and hhepf, a six-person news panel that ends
with two labels and confusion 0.36: the one file where people are merged
at scale. Offline pyannote over the whole file, with its own VBx
clustering and all ten minutes of context, splits the under-counted files
in two groups. On EN2002d (3 of 4) and TS3003a (1) it lands where the
streaming path does: a recording and model limit, and the thresholds are
not bent to it. On hhepf and iacod it is exact (6 of 6 at DER 0.077, 3
of 3 at 0.071): clean in-the-wild audio whose voices the model tells
apart with the whole file in hand, and which the 10 s windows plus the
cosine memory merge anyway. That is the one real pipeline under-count on
the set, and the open item this phase leaves: a panel of many
short-turn speakers in clean audio. The over-counts are the pipeline's
too: on EN2002b, ES2004b and IS1009d the offline pipeline finds exactly
four speakers where the 10 s windows mint 7, 6 and 5 labels. The over-counts are fragments on far-field meetings (EN2002b: 7
labels for 4). Confusion over the whole set, 0.103, is above the Phase 2b
target of 0.08, which was set on three meetings over their first 120 s;
at that horizon the same code still scores 0.021.

**Classroom result** (defaults, settled labels):

| question | speaker | noise | length | label | verdict |
|---|---|---|---|---|---|
| 1 | FEE016 | - | 11.3 s | spk_3 | own |
| 2 | FEE016 | - | 10.5 s | spk_3 | own |
| 3 | MEO015 | - | 8.3 s | spk_2 | **instructor** |
| 4 | MEO015 | - | 11.5 s | spk_6 | own |
| 5 | MEO015 | - | 14.8 s | spk_6 | own |
| 6 | FEE016 | - | 9.1 s | spk_3 | own |
| 7 | MEE014 | 5 dB pink | 8.1 s | spk_7 | own |
| 8 | MEE014 | 5 dB pink | 11.5 s | spk_7 | own |
| 9 | MEE014 | 5 dB pink | 9.7 s | spk_0 | own |

8 of 9 questions (89 percent) get a label other than the instructor's,
including all three noisy ones; 7 percent of question seconds are
labelled as the instructor; the one miss is an 8 s question whose track
scored above the match threshold against the instructor's centroid. The
instructor's label covers 96.5 percent of the lecture time. The cost is
on the other side: 8 labels for 4 people, because the instructor's first
seconds mint two labels before the centroid settles, two more are
fragments of 0 and 2 s, and two questioners get a second label on one of
their questions (only one questioner keeps a single label throughout).
Settled DER 0.105, confusion 0.060.

**Diarization period.** `diarization_period_ms` was already a provider
setting (default: the caption period, 5000; the reference config sets it
explicitly). At 6000 in the container (`caption_latency.py
--provider-set diarization_period_ms=6000`, 180 s of ES2004a) the
diarization worker's RTF is 0.249 against 0.299 at 5000, within the 0.25
target, but the label latency after the text was shown has a p95 of
12.7 s against the 4 s target (p50 0 s; every label still arrives with
the final, so nothing is corrected afterwards), 96.1 percent of final
words got a label (must: 97), and on the dev set the settled DER rises
from 0.279 to 0.296 because consecutive windows share 4 s instead of 5.
The period therefore stays at 5 s; a deployment that needs the CPU
margin more than the second of label latency can set 6000.

**Container run and targets.** One run of the standard set in the Linux
CPU reference container (`make benchmark_diarization_gate_docker
SET=standard STREAM_SEC=0 SUITE_ARGS=--skip-offline`; the whole-file
offline pass is deterministic and comes from the native
`benchmark_diarization_offline_counts` run). Acceptance
(`TARGETS=benchmarks/diarization/baselines/phase2c_acceptance.json`, with
the Phase 2a and 2b files checked as well):

| target | result |
|---|---|
| labels per real speaker between 0.8 and 1.2 | **met**: 1.17 (margin 0.03 to the upper edge; the set over-counts) |
| speaker count within one on at least 80% of meetings | **met**: 20 of 24 (83%); exact on 8 |
| classroom: at least 80% of questions get a label other than the instructor's | **met**: 8 of 9 (0.889) |
| diarization real-time factor at most 0.25 | **missed by 0.051**: 0.301 (worker 0.274 cores, 866 MB, pass 1.51 s mean). The diarization pipeline is unchanged since Phase 2b, which measured 0.277 and 0.299 on the same code path, so this is where a 10 s window at a 5 s period lands on 4 shared CPUs; the only lever that reaches 0.25, a 6 s period, costs label latency (above) and was not adopted |
| Phase 2b musts: settled DER better than 0.408; at least 97% of final words labelled | **met**: 0.267; 99.6% (230 of 231) |
| Phase 2a targets: captions within 10% of off (p50 and p95), no extra dropped periods, 1 core, 1 GB, label within 2 s p50 / 4 s p95 after text, no label change after sending, under 5% corrections | **met**: 5.71 s on against 5.82 s off (p95 22.8 against 38.6), 9 dropped periods against 13 off, 0.27 cores, 866 MB, 0 / 0 s, 0, 0 |
| Phase 2a target: diarization RTF at most 0.3 | **missed by 0.001**: 0.301 against 0.299 in the Phase 2b run of the same diarization code (run-to-run noise on the shared VM; reported, not relaxed) |
| Phase 2b target: confusion at most 0.08 | **missed by 0.023**: 0.103 on 24 meetings over their full 10 minutes (the target was set on 3 meetings over 120 s, where the same code scores 0.021); the far-field AMI meetings carry it (ES2004c 0.30, TS3003c 0.26, TS3003d 0.21) |
| Phase 2b target: settled DER at most 0.32 | **met**: 0.267 (first seen 0.271, JER 0.514) |

Replay cost in the container: tick 1.41 s mean, 1.87 s p95 against the
5 s period; label latency from a reference onset to the first pass
covering it 4.6 s p50 / 11.2 s p95 (118 of the onsets over 24 files and
four hours of audio were never covered by a pass, most of them under a
second long). The diarization-off caption run again carried the VM's
outlier (p95 38.6 s, 15 s of audio dropped) while the on run was clean
(0 s dropped). Report: `baselines/phase2c-linux-cpu-4c8g.json`
(`results/phase2c/final_linux.json`), hygiene clean before and after.

**Gate baseline.** The regression gate still compares against the Phase
2b baselines (dev set, first 120 s), which the Phase 2c report cannot be
compared with (different set and horizon: the gate run above fails on
exactly one metric, onsets never labelled, 118 against 7, for that
reason). The baseline is **not moved**: the Phase 2a RTF target is missed
by 0.001 in this run, and the rule is that every target holds before a
move. To gate against the standard set once that is settled, copy
`baselines/phase2c-linux-cpu-4c8g.json` over
`baselines/linux-cpu-4c8g.json` in its own commit; until then run the
gate with `SET=dev STREAM_SEC=120` to compare like with like. The
Phase 2b thresholds and defaults are unchanged, so the Phase 2b baseline
is still what this code produces on that set.

### Phase 2 wrap-up results

RTF headroom and classroom over-splitting, measured 2026-10-04 on top of
the Phase 2c code. Two open items from Phase 2c are addressed here: the
diarization real-time factor sat at 0.301 against the 0.3 ceiling (the gate
baseline was therefore not moved), and the classroom case minted 8 labels
for 4 people. Every replay number below comes from the cached pyannote
passes of Phase 2c (`results/phase2c/cache_*.pkl`, 10 s window, 5 s period)
through the production reconciler and attacher, so a configuration costs
seconds to score and the pyannote stage is identical across rows; the
container run at the end measures the service itself.

**Where the pass cost goes.** In the Phase 2c container run a pass cost
1.41 s, of which the embedding stage took 1.34 s (95 percent), segmentation
0.13 s and clustering 0.002 s. pyannote's pipeline extracts one embedding
per (chunk, speaker slot): for community-1 that is three slots per 10 s
window, inactive slots included, and each one is a full forward pass of the
WeSpeaker ResNet34 over the same 10 s waveform, because the speaker mask
only enters the network's final statistics-pooling layer. The context now
runs the network's frame stage once per window and pools it once per
**active** speaker track (`shared_embeddings`, on by default), and skips
the pipeline's clustering stage, whose output the local pass never used.

| | pipeline pass (Phase 2c) | shared pass (default now) |
|---|---|---|
| embedding network forward passes per window | 3 (one per slot) | 1 (frame stage) + one pooling per active track |
| native pass cost, mean over 24 ES2004a windows, 1 thread | 0.52 s (0.75 s on windows with speech) | 0.20 s (0.27 s) |
| native tick cost, dev set, 360 passes, 2 threads (cache build) | 0.709 s mean / 0.815 s p95 | 0.200 s mean / 0.243 s p95 |
| native replay tick, dev set at 120 s, suite run | - | 0.174 s mean / 0.238 s p95 (embeddings 0.126 s, segmentation 0.047 s) |
| embeddings | - | identical: largest relative difference 1.7e-6 over 468 embeddings, lowest cosine 0.9999998 |
| exclusive segments (what captions use) | - | identical in 360 of 360 passes |
| settled DER / confusion, dev set at 120 s | 0.313 / 0.021 | 0.313 / 0.021 |

Accuracy is unchanged by construction and by measurement: the dev-set
cache rebuilt with the shared pass gives the same segments and the same
embeddings to float precision, and the suite's replay on the dev set at the
gate horizon reproduces the Phase 2b numbers exactly.

*Why not a per-chunk cache.* The idea was to cache segmentation and
embeddings per audio offset and compute only the new 5 s of each window.
It cannot save anything here: the 10 s window **is** one segmentation
chunk, the segmentation model's output for [t - 10, t] is not composed from
the outputs for [t - 10, t - 5] and [t - 5, t], and the embedding is pooled
over the whole chunk. Consecutive windows share audio, not model
computation, and a cache keyed by offset would never hit at a 5 s period
because the chunk boundaries move by 5 s every tick. It would only pay if
windows were tiled without overlap (period = window), which doubles label
latency. The reuse that does exist is inside a window, across speaker
slots, and that is what the shared pass takes.

*Cheap wins checked and not taken* (native, 1 thread, frame stage over a
10 s window; `results/phase2d/onnx_quant_check.log`): batch size is moot
once the network runs once per window; skipping embeddings for inactive
slots is part of the shared pass (on the dev set 1.2 of 3 slots are active
on average, and windows without speech never reach the embedding stage);
ONNX Runtime fp32 runs the frame stage in 214 ms against 231 ms for torch
(7 percent, relative difference 5.8e-6), not worth a second inference
runtime and an export step; ONNX Runtime dynamic int8 takes 1306 ms (six
times slower on this CPU, which has no int8 kernels for these
convolutions) and moves the embeddings to cosine 0.94 against fp32;
torch's dynamic int8 quantisation touches only `Linear` layers, of which
the frame stage has none.

*A measurement fix that came with it.* The old local pass read pyannote's
`speaker_counting` hook artefact after the pipeline had finished, and the
pipeline's own `to_diarization` had by then capped that count in place to
its cluster count (one on a one-chunk window). The exclusive turns were
unaffected, but the "overlap-aware" turns the context reported were capped
to one speaker per frame as well, which is why every `overlap_*` metric of
Phase 2c equals its exclusive counterpart. The shared pass computes the
count itself, so the `overlap_aware` context option now really reports
overlapping turns. On the dev set (full 10 min, from the rebuilt cache)
the overlap-aware output scores settled DER 0.262 and confusion 0.054
against 0.279 and 0.061 for the exclusive output, with 1.08 instead of
0.92 labels minted per speaker. On the standard set (24 files, rebuilt
cache) it scores 0.258 against 0.266, the gain being missed speech (0.114
against 0.124) at the same confusion (0.102), but it mints 1.22 labels per
speaker instead of 1.17 (1.10 against 1.06 on settled captions), gets the
count within one on 18 instead of 19 meetings and revises three times as
much audio (1.9 against 0.6 percent). The default stays exclusive in this
step: words are attributed to one speaker either way, and the extra
labels cost more on screen than the DER gains.

**Merging converged speakers, re-tested.** The merge path that shipped
disabled in Phase 2b (`diarization_merge_threshold`: two speakers whose
centroids reach this cosine are merged) was swept together with a new age
limit (`diarization_merge_max_age_sec`: only a speaker minted within this
many seconds may be merged away, so two established speakers are never
joined). Thresholds 0.7 to 0.35, ages 0 (any), 10, 15, 20, 30 and 60 s:

| set | threshold 0.7 to 0.5 | 0.45 | 0.40 | 0.35, any age |
|---|---|---|---|---|
| dev + classroom: merges / confusion (0.061 without) | 0 to 1 / 0.061 | 0 to 4 / 0.061 to 0.091 | 1 to 5 / 0.076 to 0.099 | 8 / 0.107 |
| classroom labels on settled captions (8 without) | 8 | 8 | 8 | 8 |
| standard set: merges / confusion (0.103 without) | 0 to 3 / 0.103 to 0.105 | 1 to 8 / 0.105 to 0.110 | 5 to 12 / 0.107 to 0.111 | 21 / 0.113 |
| standard set: labels on settled captions per speaker (1.14 without) | 1.14 | 1.13 | 1.11 to 1.12 | 1.12 |

It never removes a classroom label: the extra labels there are not voices
that converge on an existing speaker. Below 0.5 it starts merging real
people (confusion up, speaker count within one down to 19 of 24 at 0.35
with no age limit) for at most 0.03 fewer visible labels per speaker. It
stays disabled; the age limit ships at 20 s for a deployment that enables
it (at 0.40 it halves the confusion cost of merging, 0.107 against 0.111).

**What the classroom's extra labels are.** Replaying the classroom cache
with the production reconciler and tracing every minted label
(`results/phase2d/`; the diagnostic is in the Phase 2 wrap-up log) gives
four real labels and four fragments:

| label | who | minted on | speech it got | heard again | what the next pass did with its audio |
|---|---|---|---|---|---|
| spk_0 | instructor | pass 0, the padded first 5 s window | 5.0 s | in 8 of 110 later passes | relabelled all 5 s as spk_1 (3 s) and spk_2 (2 s) |
| spk_1 | instructor | pass 1, split of the instructor | 4.0 s | never | relabelled the 1 s inside the next window as spk_2 |
| spk_2 | instructor | pass 1 | 848 s | 99 passes | confirmed |
| spk_3 | questioner FEE016 | pass 11 | 44.5 s | 8 passes | confirmed |
| spk_4 | instructor | pass 23, split of the instructor | 5.0 s | never | relabelled 4.6 of 4.6 s as spk_2 |
| spk_5 | questioner MEO015 (question 3) | pass 32 | 6.3 s | 3 passes, 0.1 s | relabelled 2.1 of 2.6 s as the instructor |
| spk_6 | questioner MEO015 | pass 43 | 50.5 s | 7 passes | confirmed |
| spk_7 | questioner MEE014 | pass 77 | 41.5 s | 7 passes | confirmed |

The fragments' centroids are far from everyone at minting (cosine 0.06 to
0.26 to the speaker they belong to; the padded first window's embedding is
unreliable) and they get no further audio to move them, so no centroid
convergence can catch them. What does identify them is the next pass: the
windows overlap by 5 s, so the audio a label was minted on is diarized
again one period later with context on both sides, and in every fragment's
case that pass labels the same audio as an existing speaker, while every
real speaker is heard again under its own label.

**Fragment fold.** That observation is now a rule in the reconciler
(`diarization_fragment_fold_sec`, default 7.5 s, and
`diarization_fragment_fold_fraction`, default 0.8): a label minted within
`fragment_fold_sec` of session time whose audio inside the current window
the current pass covers with other speakers' labels, for at least the
fraction of it, is folded into the speaker covering most of it. The fold
is reported to the attacher like a merge (`relabel`), which relabels the
fragment's spans before they settle, so the extra label never reaches a
settled caption; the fragment's label and centroid are **parked** rather
than discarded, and a voice that later matches a parked centroid takes its
old label back instead of minting a new one (this keeps the audit's six
regression cases passing: a real second voice that one window missed
returns under the label it had). 7.5 s is one and a half diarization
periods: exactly the next pass, and nothing later, may fold a label; scale
it with `diarization_period_ms`. The target's centroid is left as it is.
A fragment the next window does not reach is left alone (no evidence
either way).

Sweep (`configs/tune_grid_fold*.json`, fold window 5.5 s = next pass,
10.5 s = next two passes, fractions 0.5 to 1.0):

| configuration | classroom labels minted / on settled captions (4 people) | questions with own label | dev + classroom settled DER / confusion | standard set: settled DER / confusion | labels on settled captions per speaker | settled count within one / exact |
|---|---|---|---|---|---|---|
| no fold (Phase 2c) | 8 / 8 (2.0 per person) | 8 of 9 | 0.231 / 0.061 | 0.267 / 0.103 | 1.14 | 20 of 24 / 10 of 24 |
| **next pass, fraction 0.8 (default)** | **8 / 6 (1.5 per person)** | **8 of 9** | **0.228 / 0.058** | **0.266 / 0.102** | **1.06** | **19 of 24 / 11 of 24** |
| next pass, fraction 0.5 | 8 / 6 | 8 of 9 | 0.228 / 0.058 | 0.267 / 0.103 | 1.06 | 19 / 11 |
| next pass, fraction 1.0 | 8 / 6 | 8 of 9 | 0.228 / 0.058 | 0.266 / 0.102 | 1.10 | 19 / 10 |
| next two passes, fraction 0.5 (no parking) | 11 / 7 | 8 of 9 | 0.232 / 0.062 | 0.267 / 0.103 | 1.02 | 22 / 12 |

The classroom ends with 6 labels on settled captions for 4 people (1.5
per person, the target's edge): spk_0, spk_1 and spk_4 fold into the
instructor and spk_5 into the instructor as well (question 3 was already
scored as the instructor's), 8 of 9 questions keep a non-instructor label
and the instructor's label covers 97.7 percent of the lecture time (96.5
before). The two labels left over are confirmed voices: an instructor
fragment the next window heard again for 0.4 s (spk_5 in the folded run,
6.9 s on settled captions) and the noisy questioner's second label on
question 9 (cosine 0.28 to that questioner's first label, just under the
new-speaker threshold). On the dev set nothing folds (ES2004a's fifth label
was already off settled captions). On the standard set the fold changes
six files: EN2002a and EN2002b lose 0.004 to 0.005 DER, jwggf 0.008 and
its confusion halves, uicid (one speaker) and ylgug (two) end with the
right count, and **epygx**, a five-person debate, drops from 5 to 3
visible labels (DER 0.196 to 0.204): two real speakers whose first turns
the next window re-labelled as another panelist, the known cost of the
rule. Net on 24 files: settled DER 0.267 to 0.266, confusion 0.103 to
0.102, labels on settled captions per speaker 1.14 to 1.06 (band 0.8 to
1.2), exact count 10 to 11 of 24, count within one 20 to 19 of 24 (79
percent, so the Phase 2c "80 percent" target is missed by one meeting,
epygx). Labels minted per speaker stays 1.17 either way: the fold does not
change what is minted, only what reaches the screen, which is why the
labels-per-speaker metrics now come in two kinds (below).

In the live service the fold arrives one period after the fragment's
label was computed, so a fragment label can still reach an **in-progress**
caption for up to a period and be corrected on finalization; a label
already sent on a finalized caption is never changed (the attacher's rule),
so a fragment whose caption finalized within that period stays. The
container run measures both (`label_changes_before_final_fraction`, target
at most 5 percent, and `label_changes_after_sent`, must be 0).

**Two kinds of label count.** Since this step every count metric exists
twice: on the labels the reconciler minted or used in any pass, and on the
labels present on **settled captions** (what a viewer ends up seeing; a
folded fragment is in the first and not the second). The acceptance band
0.8 to 1.2 and the gate now use the settled-caption versions
(`replay.labels_settled_per_reference_speaker`,
`replay.speaker_count_settled_within_1_fraction`,
`replay.speaker_count_settled_exact_fraction`,
`classroom.labels_per_person_settled`); the minted and used versions stay
in the report. And, as asked, every count is reported both against the
reference speakers with at least 5 s of speech and against every reference
speaker however briefly they spoke (`*_all`). Standard set, final
configuration:

| reference speakers counted | labels minted per speaker | labels on settled captions per speaker | count within one | exact |
|---|---|---|---|---|
| at least 5 s of speech (the default since Phase 2c) | 1.17 | 1.06 | 19 of 24 (settled), 20 of 24 (any pass) | 11 of 24 (settled), 8 of 24 (any pass) |
| any speech | 1.05 | - | 19 of 24 (any pass) | 10 of 24 (any pass) |

The two references differ on six files (IS1009b 3 against 4 speakers,
IS1009c 2 against 4, TS3003a 2 against 4, TS3003b 3 against 4, jwggf 3
against 5, ylgug 2 against 3): people with under 5 s in ten minutes, who
cannot earn a label at a 2.5 s minting minimum.

**Offline reference (the model's ceiling), 9-file subset.** The
whole-file offline pass exists for 9 of the 24 files: the three dev
meetings and the six files Phase 2c chose because the streaming count
was wrong or most inflated. The remaining 15 were not run (about 90 min
of compute), so this is a **biased subset leaning to the hard files**,
not the set. Offline is the community-1 pipeline over the whole 10 min
(VBx clustering with all the context); streaming is the final
configuration replayed from the cached passes, settled labels; aggregates
pooled by reference speech (4770 s). `results/phase2d/offline_subset_comparison.json`.

| file | people | offline DER / confusion / missed | streaming DER / confusion / missed | speakers offline / streaming (settled) |
|---|---|---|---|---|
| ES2004a | 4 | 0.374 / 0.032 / 0.288 | 0.412 / 0.105 / 0.233 | 4 / 4 |
| ES2004b | 4 | 0.176 / 0.014 / 0.100 | 0.200 / 0.030 / 0.095 | 4 / 6 |
| IS1009a | 4 | 0.267 / 0.062 / 0.159 | 0.290 / 0.078 / 0.153 | 4 / 4 |
| IS1009d | 4 | 0.124 / 0.018 / 0.070 | 0.303 / 0.194 / 0.064 | 4 / 5 |
| TS3003a | 2 | 0.159 / 0.010 / 0.132 | 0.158 / 0.008 / 0.118 | 1 / 1 |
| EN2002b | 4 | 0.422 / 0.038 / 0.359 | 0.556 / 0.182 / 0.355 | 4 / 7 |
| EN2002d | 4 | 0.401 / 0.071 / 0.311 | 0.411 / 0.079 / 0.305 | 3 / 3 |
| VoxConverse hhepf | 6 | 0.077 / 0.013 / 0.024 | 0.422 / 0.360 / 0.019 | 6 / 2 |
| VoxConverse iacod | 3 | 0.071 / 0.004 / 0.022 | 0.155 / 0.087 / 0.012 | 3 / 2 |
| **pooled, 9 files** | | **0.243 / 0.031 / 0.175** (false alarm 0.036) | **0.338 / 0.129 / 0.164** (false alarm 0.045) | |

Streaming confusion is **0.098 above** the offline pipeline on these
files, far outside the 0.02 that would make it a model limit, so it is
not treated as one: missed speech and false alarm match offline (the
segmentation is the same model), and the whole gap is identity, that is
clustering 10 s windows against a cosine memory instead of the whole
file. Where the streaming count is right (ES2004a, IS1009a, EN2002d,
TS3003a) the gap is 0.00 to 0.07; where it is wrong it is 0.14 to 0.35
(IS1009d one extra label, EN2002b three, hhepf four people merged). This
is the open item for a later phase, with the subset caveat: on the 15
files not run, several of which streaming already counts exactly, the
gap is expected to be smaller.

**Container run.** One run of the standard set in the Linux CPU
reference container (`make benchmark_diarization_gate_docker
SET=standard STREAM_SEC=0 SUITE_ARGS=--skip-offline`, hygiene clean before
and after), then one repeat of the caption-latency step alone, because the
first run's diarization-on caption stream was an outlier of the kind the
audit's section 1.4 and the Phase 2b run describe (Whisper's own
executions on the caption worker averaged 11.5 s against 5.2 s in the off
stream minutes earlier, while the diarization worker sat at 0.12 cores).
Both are reported; the reports are
`baselines/phase2-wrapup-linux-cpu-4c8g.json` (the suite run) and
`baselines/phase2-wrapup-caption-repeat-linux-cpu-4c8g.json`.

| metric | Phase 2c run | wrap-up run | wrap-up caption repeat |
|---|---|---|---|
| diarization RTF / pass cost mean | 0.301 / 1.51 s | **0.135 / 0.67 s** | **0.132 / 0.66 s** |
| diarization worker cores / peak RSS | 0.274 / 866 MB | **0.124 / 707 MB** | 0.122 / 702 MB |
| diarization passes / dropped periods / audio skipped | 36 / 0 / 0 | 36 / 0 / 0 | 36 / 0 / 0 |
| diarization lag p95 | - | 1007 ms | 933 ms |
| replay tick mean / p95 / worst (24 files) | 1.41 / 1.87 / 10.7 s | **0.58 / 0.82 / 2.1 s** | - |
| replay stage: embeddings / segmentation / other | 1.34 / 0.13 / 0.002 s | **0.44 / 0.14 / 0.002 s** | - |
| replay peak RSS | 886 MB | 728 MB | - |
| settled DER / confusion / JER | 0.267 / 0.103 / 0.514 | 0.266 / 0.102 / 0.512 | - |
| labels minted / on settled captions per speaker | 1.17 / 1.14 | 1.17 / 1.06 | - |
| settled count within one / exact | 20 / 10 of 24 | 19 / 11 of 24 | - |
| classroom labels on settled captions / own-label questions | 8 / 8 of 9 | **6 / 8 of 9** | - |
| caption p50, chunk-id in-progress, on / off | 5.71 / 5.82 s | 10.45 / 5.52 s | 4.92 / 4.39 s |
| caption p95, on / off | 22.8 / 38.6 s | 41.4 / 21.7 s | 28.4 / 22.1 s |
| caption periods dropped (of 36), on / off | 9 / 13 | 19 / 10 | 10 / 9 |
| Whisper execution mean / p95 on the caption worker, on | - | 11.5 / 26.7 s | 4.7 / 20.8 s |
| Whisper execution mean / p95, off | - | 5.2 / 31.6 s | 4.4 / 19.4 s |
| finalized words labelled | 99.6% | 98.1% | 97.2% |
| label after text p50 / p95 | 0 / 0 s | 0 / 0 s | 0 / 0 s |
| label corrections before final / changes after sent | 0 / 0 | 0.8% / 0 | 0.4% / 0 |

Acceptance, on the suite run with the caption columns of both runs:

| target | result |
|---|---|
| wrap-up: diarization RTF at most 0.25 | **met**: 0.135 (margin 0.115; the pass costs 0.67 s of a 5 s period, down from 1.51 s) |
| Phase 2a: RTF at most 0.3, at most 1 core, at most 1 GB | **met**: 0.135, 0.12 cores, 707 MB |
| wrap-up: classroom at most 1.5 labels per person on settled captions | **met at the edge**: 1.5 (6 for 4) |
| wrap-up and 2c: at least 80% of classroom questions with their own label | **met**: 8 of 9 |
| wrap-up: labels on settled captions per speaker in 0.8 to 1.2 | **met**: 1.06 (2c's minted ratio 1.17 also inside the band) |
| wrap-up: settled count within one on at least 80% of meetings | **missed by one meeting**: 19 of 24 (79 percent), epygx; on the minted and used counts (2b, 2c) 20 of 24, met |
| Phase 2b musts: settled DER better than 0.408, at least 97% of final words labelled, the six regression tests | **met**: 0.266; 98.1% (97.2% in the repeat); 114 reconciler and job tests pass |
| Phase 2b / 2c: confusion at most 0.08 | **missed by 0.022**: 0.102 (unchanged from 2c; the offline reference above puts 0.10 of it on the clustering context) |
| Phase 2a: label within 2 s p50 / 4 s p95 after the text, no change after sending, under 5% corrections | **met**: 0 / 0 s, 0, 0.8% (a folded fragment label that had reached an in-progress caption) |
| Phase 2a: caption p50 and p95 with diarization on within 10% of off, no more dropped periods than off | **missed in both single runs, met on the three alternating pairs** (next section): suite run 10.45 against 5.52 s (+89%), 41.4 against 21.7 s, 19 against 10 dropped; repeat 4.92 against 4.39 s (+12%), 28.4 against 22.1 s (+28%), 10 against 9 dropped; pairs, median on / off ratio: p50 0.86, p95 1.01, dropped 0.75 |

What to make of the caption miss. The diarization worker now costs a
tenth of a core, half of what it did when these targets were met in
Phase 2a, 2b and 2c (on 5.71 against off 5.82 s in 2c), and the caption
worker runs upstream's unchanged job; both streams in the repeat show the
same Whisper execution profile (mean 4.7 against 4.4 s, p95 20.8 against
19.4 s) and the first run's on stream shows Whisper itself running twice
as slowly for three minutes. The run-to-run spread of the same off
configuration in this step alone (p50 5.52 against 4.39 s, p95 21.7
against 22.1 s, 10 against 9 drops) is larger than the on-against-off
difference of the repeat, so these two runs cannot attribute the
difference to diarization, and they cannot rule it out either. The target
is reported as missed, not explained away.

**Caption parity, three alternating pairs.** Single runs cannot separate
noise from contention (the off configuration alone moved from 4.39 to
5.52 s p50 between the two runs above), so the caption-latency step was
repeated as three alternating pairs, off, on, off, on, off, on, in the same
container with the same clip and settings (ES2004a, 180 s, reference
config), hygiene clean before every run. Report:
`baselines/phase2-wrapup-caption-pairs-linux-cpu-4c8g.json` (the six
caption reports with a summary). Whisper's execution time is the
`asrExecutionMs` histogram of the caption worker at the end of each run.

| run | caption p50 | caption p95 | periods dropped (of 36) | Whisper execution mean / p95 | diarization worker cores |
|---|---|---|---|---|---|
| pair 1 off | 5.97 s | 28.3 s | 11 | 5.36 / 23.2 s | - |
| pair 1 on | 6.02 s | 28.8 s | 11 | 5.67 / 33.7 s | 0.116 |
| pair 2 off | 5.88 s | 28.1 s | 9 | 4.77 / 13.9 s | - |
| pair 2 on | 4.67 s | 28.3 s | 6 | 3.46 / 7.1 s | 0.115 |
| pair 3 off | 6.90 s | 39.4 s | 16 | 7.49 / 60.2 s | - |
| pair 3 on | 5.90 s | 28.4 s | 12 | 5.57 / 28.2 s | 0.121 |
| on / off ratio per pair, median | **0.86** (1.01, 0.79, 0.86) | 1.01 (1.02, 1.01, 0.72) | 0.75 (1.00, 0.67, 0.75) | mean **0.74** (1.06, 0.73, 0.74); p95 0.51 | |
| median of the three runs per mode, off / on | 5.97 / 5.90 s | 28.3 / 28.4 s | 11 / 11 | 5.36 / 5.57 s; p95 23.2 / 28.2 s | |

Decided by the median across the pairs: the on / off p50 ratio is 0.86
(within 10 percent) and Whisper's own execution time is not higher with
diarization on (paired ratio 0.74; in two of the three pairs Whisper ran
faster beside the diarization worker than alone). Taking the median run of
each mode instead gives p50 0.99 and Whisper mean 1.04, p95 1.22; that 4
percent sits inside the off configuration's own spread (4.77 to 7.49 s
mean across its three runs), and the pairs that drive it go the other
way. Parity **holds**; the two single runs above were noise, the first one
an outlier of the kind the audit describes. The settings that keep the
diarization worker off Whisper's cores were confirmed in the container
rather than assumed: the image pins `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, the context sets torch's intra-op pool to its
`num_threads` of 1 on every pass (the inter-op pool is left at the
default and is unused by this inference), the diarization worker logs
`Lowered diarization worker scheduling priority (nice 10)`, and it sat at
0.12 cores in every on run while Whisper's CTranslate2 ran with
`cpu_threads: 4`. No change was made to the service for this result.

**Gate baseline: moved.** The condition for moving it was every Phase 2a
and 2b must target holding, the RTF ceiling included, in the container
run. With caption parity settled by the pairs above it holds, and
`baselines/linux-cpu-4c8g.json` is now a copy of
`baselines/phase2-wrapup-linux-cpu-4c8g.json` (its own commit, before and
after numbers in the message), so `make benchmark_diarization_gate_docker`
compares like with like at its defaults (standard set, full 10 min). Two
caveats travel with it: the baseline's caption-on columns are the outlier
stream of that run (p50 10.45 s), so the gate's `caption.on` tolerances
are looser than the service's real behaviour until a suite run with a
clean caption stream replaces the file, and the offline metrics are
absent (the run skipped them), so `replay.offline.*` is skipped by the
gate rather than compared. The Phase 2b baselines stay in `baselines/`
for the dev set at 120 s.

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

Diarization costs one pass per period on its own worker. Up to Phase 2c a
10 s window cost about 1.4 s per 5 s period on the 4-CPU reference
container (RTF 0.30) at one torch thread, 95 percent of it in the
embedding network, which pyannote ran three times per window; since the
Phase 2 wrap-up the network runs once per window: on the same container a
pass costs 0.67 s (RTF 0.135), the worker 0.12 cores and 0.7 GB. See
"Phase 2a results" for how many sessions the container sustains. The
caption worker is never slowed by it: if the diarization
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
| `transcription_service` | `PyannoteDiarizationContext`: optional worker-pool context running `pyannote/speaker-diarization-community-1` (CPU by default, CUDA-ready) with its own thread cap, nice and segmentation step; on one-chunk windows it reports the segmentation's local speaker tracks and runs the embedding network once per window (a third of the pipeline's pass cost, same embeddings) |
| `transcription_service` | `DiarizationJob`: a second worker-pool job per session on the diarization worker, newest-window passes, skip-based back-pressure, lag and uncovered-audio counters |
| `transcription_service` | `SpeakerReconciler`: maps per-pass labels to session-wide labels by speaker-embedding memory plus overlap voting, so one voice keeps one label across passes; folds a fresh label the next pass re-labels as an existing speaker before it settles (parking it for revival) |
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

Plain-language notes for deployers, each with its measured size (AMI
far-field meetings and VoxConverse panels, first 10 minutes, 10 s window,
5 s period; "Phase 2c results" and "Phase 2 wrap-up results" hold the
tables).

- **Far-field rooms under-count quiet or brief speakers.** With one
  distant microphone, a person who speaks little may never get a label:
  TS3003a's second speaker (11.6 s in ten minutes) stays inside the main
  speaker's label, and EN2002d ends with 3 labels for 4 people. Offline
  pyannote over the whole file lands in the same place on both, so this is
  the model and the recording, not the streaming path. On the 24-file set,
  5 meetings end with fewer labels on screen than people who spoke 5 s or
  more, 8 with more.
- **Clean panels of many short-turn speakers collapse in streaming.**
  Where offline pyannote separates every voice with the whole file in
  hand, the 10 s windows plus the session memory merge people: the
  six-person news panel hhepf ends with 2 labels (confusion 0.36 of its
  speech) and the three-person iacod with 2. This is the one real
  streaming-pipeline under-count and it is open.
- **The fragment fold trades a debate for the classroom.** The fold
  (`diarization_fragment_fold_sec`, default 7.5 s; 0 disables) removes a
  label the next window re-labels as an existing speaker. In a classroom
  that is a split of the instructor; in the five-person debate epygx it
  was two real speakers whose first turns the next window put under
  another panelist, so epygx shows 3 labels instead of 5 (settled DER
  0.196 to 0.204) and the set's count-within-one goes from 20 to 19 of 24.
  It is on by default because the classroom is the product case and DER,
  confusion and the label band hold on the set; a deployment with
  many-speaker panels and few fragments can set it to 0.
- **Streaming identity is about 0.10 confusion behind the offline
  pipeline.** On the 9 files that have a whole-file offline pass (a subset
  leaning to the hard files) offline pyannote scores confusion 0.031 and
  DER 0.243 against streaming's 0.129 and 0.338; missed speech and false
  alarm match, so the gap is the clustering context (10 s windows plus a
  cosine memory against the whole file), not the model. See "Phase 2
  wrap-up results".
- **One person can still appear as two labels.** The typical far-field
  error is a split, not a merge: 1.17 labels are minted per real speaker
  on the set and 1.06 reach settled captions (1.14 before the fold); 11 of
  24 meetings end with exactly the right count, 19 within one. The
  classroom case ends with 6 labels for 4 people: an instructor fragment
  the next window heard again and a second label for the noisy
  questioner. A new label is minted only after about 2.5 s of a voice, so
  a person's first words attach to the nearest known speaker or stay
  unlabelled.
- **Wrong-person labels on far-field audio.** Speaker confusion is 0.10 of
  reference speech over the set (ES2004c 0.30, TS3003c 0.26, TS3003d 0.21;
  the clean VoxConverse files 0.00 to 0.09): in a noisy far-field meeting
  about one word in ten carries another participant's label. The
  offline-pipeline ceiling on the same files is reported in "Phase 2
  wrap-up results" as it is measured.
- **Heavy noise merges voices.** Pink noise at 5 dB signal-to-noise on
  the hard-case clip turns four voices into two labels (settled DER 0.569
  against 0.392 for the clean clip). In the classroom case the questioner
  under the same noise still gets a non-instructor label on all three
  questions, but one of them under a second label.
- **Overlapping speech shows one speaker.** Words spoken over someone
  else take the dominant speaker's label, and missed speech (0.12 of
  reference speech on the set, mostly overlap and far-field pauses) is
  the segmentation model's, with no threshold to lower. The context can
  report overlapping turns (`overlap_aware`); that output scores settled
  DER 0.258 against 0.266 on the 24-file set (0.262 against 0.279 on the
  dev set) but mints more labels (1.22 against 1.17 per speaker), so it is
  not the default.
- **A fragment label can flash on an in-progress caption.** The fold
  that removes a split label arrives one period after it was minted, so
  such a label can show on the in-progress tail for up to 5 s and is
  corrected at finalization; a label already sent on a finalized caption
  never changes, by design, so a fragment whose caption finalized inside
  that period stays. The first window of a session (5 s, padded to the
  model's 10 s) gives the least reliable embedding and is the usual
  source of such a label.
- **Labels are frozen once final.** Words already finalized keep the
  label they were finalized with; later, better speaker evidence revises
  only words not yet finalized.
- **Clock shift after a caption-side audio drop.** After the caption
  worker drops audio (its buffer was full for longer than it holds), the
  drop is placed at the newest Whisper time the session had seen, so words
  between the drop and the next result can be a few seconds off against
  the diarization clock.
- **Capacity.** The 4-CPU reference container sustains one diarized
  session, limited by Whisper, not diarization (see "Phase 2a results");
  the diarization worker falls behind only at four sessions and then skips
  audio (shown unlabelled, counted) rather than delaying captions. GPU
  deployments can set `"device": "cuda"` in the context config; a
  streaming-native alternative (NVIDIA Streaming Sortformer) can be added
  as another context implementation without touching the provider wiring.
