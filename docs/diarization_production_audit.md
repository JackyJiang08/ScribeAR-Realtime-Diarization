# Speaker diarization: production-readiness audit

Date: 2026-10-02. Branch `feature/speaker-diarization` at 593f390 (merge of
upstream `staging` 5f01575). Read-only audit; no code was changed. Every
number below was measured on this day on the machine described in section 0
unless it cites `benchmarks/diarization/results/post_merge_baseline.json`.

Severity scale: **P0** blocks a release to real customers, **P1** serious,
**P2** polish. Effort is a rough engineering estimate including tests.

> **Status after Phase 2a (2026-10-03).** The P0 of section 2 is closed:
> diarization no longer runs on the caption path. It is a separate
> worker-pool job on its own worker (`DiarizationJob`), the newest 10 s per
> 5 s period, with labels attached to already-shown captions through a
> `speakers_update` message and a session-side `SpeakerLabelAttacher`
> (never holding a caption). Section 2's P1 (Silero's thread count applying
> to pyannote) is closed by the context's own `num_threads`; section 7.1
> (counters never exported) is closed, with lag, skipped audio and dropped
> periods added, a sidecar rule and two Grafana panels; section 7.3's
> fail-fast tag check and speaker-bound validation are in; section 4's
> "unlabelled words read as the previous speaker" is replaced by an explicit
> `Speaker ?` slot. The upstream-side items (section 1, Whisper per-range
> calls and `silence_threshold`) are written up as issue drafts in
> `docs/upstream_issue_drafts.md`, not changed in the fork. The measured
> outcome against the Phase 2a acceptance criteria, including the accuracy
> cost of the 10 s window, is in `transcription_service/docs/speaker_diarization.md`
> ("Phase 2a results").
>
> **Status after Phase 2b (2026-10-03).** Section 3.2 is closed: the
> reconciler keeps a per-session centroid embedding per speaker (in memory
> only, dropped with the session after a reconnect grace) and matches every
> pass against the whole session, mints only for long, clearly distinct
> voices, and pools short unknown ones; the six synthetic cases are passing
> unit tests. Section 3.5's under-counting turned out to be pyannote's
> exclusive output collapsing a one-chunk window to one speaker, fixed by
> reporting the segmentation's own speaker tracks (`local_speakers`); the
> clustering threshold (`clustering_threshold`) is exposed but has no effect
> on one-chunk windows. Section 3.1 (overlap) was measured and is
> configurable (`overlap_aware`). Section 5.3 (reconnects) is closed by the
> provider's in-memory speaker memory keyed by `session_uid`. Section 4's
> unattributed state is now an empty slot once settled, never a question
> mark. Numbers, the threshold tradeoff and the remaining misses are in
> `transcription_service/docs/speaker_diarization.md` ("Phase 2b results").
> Section 5 (robustness beyond reconnects) remains Phase 2c work.

## Executive summary

1. **The 26.8 s caption latency with diarization off is upstream's streaming
   design running under the fork's development config, not fork code.** Plain
   upstream `staging` with no fork code reproduces it (30.2 s). The dominant
   cost is upstream's `vad_detector: true` path: Silero splits each 30 s
   window into 8 to 16 sub-second ranges and Whisper is called once per
   range, so one tick costs 7 to 17 s in isolation instead of 3 s. Upstream's
   shipped deployment config does not enable VAD; with that config the same
   clip shows words after 18.8 s (word method) or 7 s (upstream's chunk-id
   method). This laptop adds the worst outliers (single ticks of 25 to 39 s
   that never reproduce in isolation) through memory pressure and background
   CPU load, not through thread settings, Apple Silicon or debug builds. See
   section 1 for the evidence and section 1.5 for realistic targets.
2. **Diarization runs synchronously inside the caption tick** and costs 11.7 s
   per 30 s window on this CPU, 92 percent of it in pyannote's embedding
   stage, and the cost grows superlinearly with the window (10 s: 0.6 s,
   20 s: 5.6 s, 30 s: 11.7 s). With it on, captions appear after 84.8 s, two
   thirds of the audio chunks are dropped, and the worker holds 1.9 GB. This
   is the P0.
3. **Label quality is not production grade yet**: streaming DER is 0.41 to
   0.46 against 0.26 offline, missed speech is the largest term, the
   reconciler mints a fresh label whenever the window slides past a voice
   (shown with synthetic cases in section 3.2), overlapping speech is
   discarded by design, and on TS3003a only one of four speakers is found.
4. **Operations gaps**: the five diarization counters are never exported;
   the production Docker images do not contain pyannote; the deployment
   folder has zero mention of diarization or the HuggingFace token; a bad
   context tag produces a per-session 1011 reconnect loop instead of a
   startup error; Silero's `torch.set_num_threads(1)` applies to pyannote.
5. **Accessibility**: in the client and kiosk webapps no caption text, and
   therefore no speaker label, ever reaches the `role="log"` live region,
   because only the standalone webapp dispatches paragraph commits. This is
   inherited from upstream and matters more once speakers exist. The color
   routine can return below 4.5:1 on mid-luminance backgrounds.
6. **Licensing**: `pyannote/speaker-diarization-community-1` is CC-BY-4.0
   (attribution required) behind a gated form; nothing stores audio or
   embeddings. Section 9.

The prioritized plan is in section 11.

## 0. Method and environment

Machine: Apple M4 (4 performance + 6 efficiency cores), 16 GB RAM, macOS
15.7.8. State during the runs: load average 5.6 before starting, swap
16.9 GB of 18.4 GB in use, 64 MB of free pages, `contactsd`,
`AddressBookManager` and `mds` each consuming 30 to 60 percent of a core,
on AC power, low power mode off. This is a loaded laptop, and the runs were
serialized so they never overlapped each other.

Software: Python 3.12.11, torch 2.13.0, pyannote.audio 4.0.7, faster-whisper
1.2.1, CTranslate2 4.7.1, numpy 2.3.3 linked against Apple Accelerate (not
OpenBLAS). No `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` set in the shell;
the Docker images set both to 1.

Audio: AMI ES2004a, first 180 s, 16 kHz mono, streamed at real time in 0.5 s
SAFP frames straight at `/transcription_stream/whisper`, bypassing
node-server (the same instrument shape as upstream's `tools/asr-load`).

Latency was measured two ways on every run:

- **Word method** (the method used for the 26.8 s figure): for each
  `(end_timestamp, text)` pair, delay from the send time of the 0.5 s chunk
  containing `end` to the first message that shows the pair, in
  `in_progress` or `final`.
- **Chunk-id method** (upstream's own correlation, added in PR #124): delay
  from the send time of the newest chunk listed in `in_progress_chunk_ids`
  or `final_chunk_ids` to the arrival of that message. node-server
  aggregates exactly this into its latency percentiles.

Scripts, logs and JSON results are in this session's scratch folder and are
not committed: `caption_latency2.py`, `whisper_micro.py`, `tick_replay.py`,
`diar_profile.py`, `reconciler_cases.py`, `run_chain*.sh`.

## 1. Why captions take 26.8 s with diarization off

### 1.1 Control: plain upstream staging, no fork code

`git archive staging transcription_service` was unpacked into a scratch
folder and synced with `uv sync --frozen --extra faster-whisper --extra
silero-vad-cpu` from upstream's own lockfile. The service was run with
`.venv/bin/python src/index.py`. Fork runs used the fork tree with a scratch
provider config injected through `PROVIDER_CONFIG_PATH`.

| run | code | provider config | words first shown (word method) mean / p95 | newest-audio latency (chunk-id, in-progress) mean / p50 | finalized (word) mean | ticks run / periods elapsed / dropped | tick exec mean / max | worker CPU mean / max | worker RSS max |
|---|---|---|---|---|---|---|---|---|---|
| stg_vad_p5 | upstream staging | base, Silero VAD on, 5 s period (same as the 26.8 s run) | 30.2 s / 50.8 s | 18.1 s / 20.3 s | 54.2 s | 19 / 36 / 17 | 6.7 s / 32.5 s | 100 % / 173 % | 917 MB |
| stg_novad_p5 | upstream staging | **upstream's shipped deployment defaults** (no `vad_detector`) | 18.8 s / 32.3 s | 7.0 s / 3.6 s | 38.2 s | 29 / 39 / 10 | 3.6 s / 38.6 s | 71 % / 160 % | 916 MB |
| stg_vad_p5_omp1 | upstream staging | as stg_vad_p5 + `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (Docker image env) | 30.6 s / 52.0 s | 16.9 s / 17.4 s | 54.9 s | 18 / 36 / 18 | 7.6 s / 33.8 s | 105 % / 167 % | 914 MB |
| stg_vad_p1 | upstream staging | as stg_vad_p5 with `job_period_ms: 1000` | 29.4 s / 48.6 s | 14.6 s / 18.9 s | 52.2 s | 51 / 35 / 17 | 3.0 s / 25.5 s | 122 % / 171 % | 967 MB |
| fork_off_nopy | fork | as stg_vad_p5, diarization off, pyannote context absent | 29.5 s / 56.7 s | 18.1 s / 18.5 s | 49.6 s | 18 / 29 / 11 | 6.4 s / 35.1 s | 98 % / 170 % | 925 MB |
| fork_off_pyloaded | fork | as stg_vad_p5, diarization off, pyannote context loaded | 30.4 s / 55.2 s | 17.8 s / 18.6 s | 54.6 s | 19 / 36 / 17 | 6.6 s / 30.7 s | 98 % / 179 % | 950 MB |
| earlier run (sync doc) | fork | as fork_off_pyloaded | 26.8 s / 39.5 s | not measured | 48.0 s | 16 / 36 / about 20 | 8.0 s / 32.8 s | not measured | not measured |

Readings:

- Plain upstream under the same config gives the same number (30.2 s vs
  26.8 s, within run-to-run noise). Fork code with diarization off is
  indistinguishable from upstream (29.5 s and 30.4 s). The fork does not
  slow the caption path when diarization is off.
- Thread caps change nothing (stg_vad_p5_omp1). A 1 s period changes nothing
  (stg_vad_p1): the bottleneck is tick execution time, not scheduling.
- Upstream's shipped defaults (no VAD) are much better: 18.8 s word method,
  7 s chunk-id, and more than twice as many transcript messages.
- Every run drops periods. Half of all 5 s periods were skipped in the VAD
  runs, a quarter in the no-VAD run.

### 1.2 Where a Whisper tick goes: micro-benchmark and tick replay

Isolated `WhisperModel("base", device="cpu", cpu_threads=4)` on one 30 s
window (ES2004a 90 to 120 s), exactly the arguments the job uses
(`word_timestamps=True`, `hallucination_silence_threshold=0.01`,
`language="en"`), three repeats after warm-up:

| variant | wall per 30 s window | CPU per call | cores |
|---|---|---|---|
| fork venv, as shipped (cpu_threads 4, float32) | 2.66 s | 3.6 s | 1.35 |
| staging venv, same | 2.61 s | 3.6 s | 1.37 |
| cpu_threads 1 | **1.86 s** | 1.85 s | 0.99 |
| cpu_threads 8 / 10 | 2.62 s / 2.44 s | 4.5 s / 4.4 s | 1.7 / 1.8 |
| compute_type int8 | 2.00 s | 6.7 s | 3.3 |
| `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, cpu_threads 4 | 2.76 s | 3.8 s | 1.37 |
| torch imported in the same process | 2.77 s | 3.8 s | 1.36 |
| with a 200-character `initial_prompt` | 3.26 s | 4.5 s | 1.38 |
| window split into 3 / 6 pieces, one call each | 3.01 s / 4.60 s | 4.2 s / 6.5 s | 1.4 |
| 10 s window | 0.88 s | 1.3 s | 1.4 |

Whisper base needs about 2.6 s for 30 s of audio here, and 1 thread is faster
than 4 (the M4's efficiency cores slow the pool down). None of the laptop
hypotheses move this number: thread caps, torch import, Apple Silicon (there
is no acceleration path; CTranslate2 has no Metal backend, confirmed by
upstream's `APPLE-SILICON-FINDINGS.md`, and the arm64 wheels are release
builds, not debug builds).

Tick replay: `_transcribe_audio` reproduced outside the service with
upstream's own `SileroVadContext` stream and `FasterWhisperContext`, on eight
30 s windows of the same clip, with the previous window's words as
`initial_prompt`:

| window | Silero ranges | per-range Whisper calls total | words | single call, whole window (as upstream's default config does) | single call without `hallucination_silence_threshold` |
|---|---|---|---|---|---|
| 0 to 30 s | 8 | 11.5 s (one 0.4 s range took 7.1 s) | 18 | 3.3 s | 2.0 s |
| 10 to 40 s | 8 | 8.4 s | 30 | **9.4 s** | 2.6 s |
| 20 to 50 s | 8 | 12.8 s (one 0.3 s range took 5.8 s) | 13 | 5.3 s | 2.7 s |
| 90 to 120 s | 14 | 13.2 s | 64 | 2.8 s | 2.8 s |
| 100 to 130 s | 13 | 16.9 s (one range 7.1 s, one 3.8 s) | 44 | 3.1 s | 3.0 s |
| 115 to 145 s | 15 | 14.4 s | 50 | 2.9 s | 3.0 s |
| 120 to 150 s | 16 | 10.5 s | 60 | 3.2 s | 3.2 s |
| 145 to 175 s | 13 | 7.3 s | 77 | 3.2 s | 3.1 s |

Readings:

- **The VAD path multiplies the cost by 3 to 5.** Each Silero range becomes a
  separate `whisper.transcribe` call, and Whisper pads every call to a 30 s
  encoder window, so a 0.5 s range costs at least 0.45 s. Individual
  sub-second ranges occasionally cost 2 to 7 s (temperature fallback and the
  silence-skip logic on tiny clips). This is upstream code
  (`whisper_streaming_job.py:563-581`), active only when a config sets
  `vad_detector: true`, which the fork's local dev config did and upstream's
  deployment template does not.
- **`hallucination_silence_threshold=0.01`** (upstream's `silence_threshold`
  default) makes faster-whisper re-seek and re-decode around any pause longer
  than 10 ms. On sparse windows it turns a 2.6 s call into 5 to 9 s. Without
  it every window costs 2.0 to 3.2 s. This affects upstream's default config
  too.
- In isolation no tick exceeds 17 s, yet the live runs show single ticks of
  25 to 39 s (including one 38.6 s single-call tick that produced one short
  sentence). That residual is the laptop: 17 GB of swap in use and three
  system daemons competing for the four performance cores. On a dedicated
  server the outliers should sit near the isolated numbers.

### 1.3 Why a slow tick becomes 30 s of caption latency

A word must wait for the tick that is running when it arrives to finish, then
for the next tick to transcribe it, then LocalAgree (`local_agree_dim: 2`)
needs a second agreeing pass before it can commit, and `final` is only
emitted at a sentence end or when the 30 s buffer force-finalizes. With ticks
of 10 to 30 s the first-shown delay is roughly two tick durations, which is
what both tables show. The chunk-id method reads lower (7 to 18 s) because
it measures the newest audio in each message; the word method is harsher
because it also counts every word that was re-timestamped by a later pass as
a new word. Both are valid; the chunk-id figure is the one comparable to
upstream's node-server percentiles.

### 1.4 Verdict on the 26.8 s

| candidate cause | verdict | evidence |
|---|---|---|
| fork code | no | fork with diarization off equals plain staging (29.5 / 30.4 vs 30.2 s) |
| thread settings (OMP, OpenBLAS, CTranslate2 threads) | no | caps change nothing; numpy uses Accelerate; 1 thread is faster than 4 |
| no Apple Silicon acceleration | no | CPU-only everywhere; upstream's own findings say no Metal path exists; the question is the tick design, not the device |
| debug builds | no | identical release wheels in both venvs, identical micro-benchmark |
| CPU contention and memory pressure on this laptop | partly: the worst outliers | isolated max 17 s vs live max 39 s; load 5.6, swap 17 GB |
| measurement method | partly: the word method roughly doubles the chunk-id figure | 30.2 vs 18.1 s; 18.8 vs 7.0 s on the same runs |
| upstream itself | **mainly** | per-range Whisper calls under `vad_detector: true` (3 to 5x), `hallucination_silence_threshold` re-decoding, 30 s re-transcription every tick, two-pass LocalAgree, 5 s period |

### 1.5 Realistic caption-latency targets for Phase 2 (CPU, Whisper base)

Measured on upstream's shipped deployment defaults (no VAD), this laptop:

| metric | now (upstream, diarization off) | achievable with the upstream-side fixes in section 11 (estimate) |
|---|---|---|
| newest audio to in-progress caption (chunk-id) | mean 7.0 s, p50 3.6 s | 3 to 5 s |
| word first shown (word method) | mean 18.8 s, p95 32 s | 8 to 12 s |
| word finalized | mean 38 s | 15 to 25 s (LocalAgree + sentence-end bound) |
| ticks dropped | 10 of 39 | 0 |

Diarization must fit inside those budgets, so its per-tick cost must come
down from 11.7 s to under about 1 s, or it must leave the caption path.

## 2. Architecture: diarization on the caption path

**P0. Diarization runs synchronously inside every caption tick.**
`whisper_streaming_job.py:560` calls `_detect_speaker_ranges` inside
`_transcribe_audio`, before the Whisper loop, on the same worker process and
the same 30 s window. Everything downstream (LocalAgree, `final`, the
WebSocket send) waits for it. Measured effect (diarization on, same clip):

| metric | diarization off (upstream defaults) | diarization on (fork_on run) |
|---|---|---|
| transcript messages in 180 s | 29 | 7 |
| words first shown, mean / p95 | 18.8 s / 32 s | **84.8 s / 104 s** |
| newest audio to caption (chunk-id) | 7.0 s | 57 s |
| tick execution, mean / max | 3.6 s / 38.6 s | 25.5 s / 76 s |
| audio chunks dropped as "buffer had no room" | 10 | 60 (30 s of speech lost) |
| worker RSS | 0.9 GB | **1.9 GB** |
| worker CPU | 0.7 core | 1.0 core |

**Where the diarization tick goes** (pyannote hook timing, CPU, ES2004a
90 to 120 s, after warm-up):

| window | wall | segmentation | embeddings | clustering (VBx) | CPU seconds |
|---|---|---|---|---|---|
| 30 s, 4 torch threads | 11.7 s | 0.9 s | **10.8 s** | about 0 | 37.8 |
| 30 s, 1 thread | 12.7 s | 1.0 s | 11.7 s | 0 | 35.5 |
| 30 s, 8 / 10 threads | 11.4 s / 11.5 s | 0.9 s | 10.4 s | 0 | 40 / 42 |
| 20 s | 5.6 s | 0.2 s | 5.4 s | 0 | 18.6 |
| 10 s | **0.58 s** | 0.05 s | 0.53 s | 0 | 1.3 |
| 5 s | 0.62 s | 0.07 s | 0.55 s | 0 | 1.3 |

Readings: 92 percent of the tick is the embedding model, not segmentation or
clustering. Cost is superlinear in window length because the pipeline runs
its 10 s segmentation window with a 1 s step (`segmentation_step` defaults
to 0.1 of the window) and extracts one embedding per window per local
speaker: a 30 s buffer means 21 windows, a 10 s buffer means one. The
reconciler itself costs microseconds. Thread count barely matters on this
machine (the embedding batch is already parallel through Accelerate), so
thread tuning is not the fix; window length is. This also explains the
baseline's 12.8 s mean tick: it is the 30 s window, not the CPU.

Fix options, in order of leverage:

1. Diarize only the newest 10 to 15 s each tick (0.6 to 2 s) and keep the
   reconciler for continuity, instead of the full transcribe window
   (`whisper_streaming_job.py:542-560`). Effort 1 to 2 days including the
   benchmark rerun.
2. Raise `segmentation_step` (pipeline parameter) to 0.3 to 0.5 so a 30 s
   window has 5 to 8 embedding windows. Half a day, accuracy to be
   re-measured.
3. Move diarization off the caption path: run it as its own worker-pool job
   on a longer period (for example 10 s) and attach labels to words
   asynchronously through a `speakers` update message, so captions never
   wait. 3 to 5 days, touches the wire format and the store.
4. Replace the per-tick full pipeline with incremental embedding extraction
   plus an online clusterer (the embedding-memory follow-up already listed in
   `docs/speaker_diarization.md`). 1 to 2 weeks.

**P1. `torch.set_num_threads(1)` from the Silero context applies to pyannote.**
`silero_vad_context.py:122` sets it process-wide inside `create()`, and
contexts are created in config order in the same worker
(`worker_process.py:196`), so pyannote inference runs with torch at one
thread on Linux; the Docker images also pin `OMP_NUM_THREADS=1`. On this
macOS machine the measured effect was small because Accelerate parallelizes
underneath torch, so the Linux effect is unverified but likely larger. Fix:
a `num_threads` field on `PyannoteDiarizationContextConfig`, set around
`diarize()` and restored afterwards. 2 hours plus a Linux measurement.

**P1 (upstream). Per-range Whisper calls under `vad_detector: true`** and
**P2 (upstream) `hallucination_silence_threshold=0.01`**, both quantified in
section 1.2. Worth raising upstream: transcribe the whole window once and use
the Silero ranges only to mask silence, or merge ranges with gaps under 1 s;
and make `silence_threshold` default to a value near 2 s or `None`. 1 day
each, upstream-side.

## 3. Accuracy

Baseline numbers (`post_merge_baseline.json`, three AMI meetings, single
distant microphone):

| | DER | missed | false alarm | confusion | JER |
|---|---|---|---|---|---|
| offline, whole 10 min | 0.260 | 0.188 | 0.038 | 0.034 | 0.52 |
| streaming, label first seen | 0.462 | 0.217 | 0.112 | 0.133 | 0.68 |
| streaming, label settled two ticks later | 0.408 | 0.208 | 0.106 | 0.095 | 0.62 |

Per file: ES2004a streaming first-seen DER 0.63 (false alarm 0.23), IS1009a
0.53 (confusion 0.34, 7 labels minted for 4 speakers), TS3003a 0.29 with
**one hypothesis speaker for four reference speakers** and JER 0.81.

### 3.1 Overlapping speech. P1.

`pyannote_diarization_context.py:65-69` prefers
`exclusive_speaker_diarization`, which assigns each instant to one speaker,
so the second speaker in an overlap is dropped before attribution. The
baseline scores overlap (pyannote convention), which is one reason missed
speech is the largest term (0.19 offline, 0.21 streaming). Words spoken
during overlap get the dominant speaker's label or `null`. Fix: use
`speaker_diarization` (overlap-aware) and let `_assign_speaker` pick the
largest overlap, which it already does; measure the DER change. Half a day.

### 3.2 Short turns, backchannels, drift, swaps, leave and return. P1.

`SpeakerReconciler` (`speaker_reconciler.py:50-98`) matches each run only to
the previous run by overlap voting and mints a new label for any raw label
with no overlapping predecessor. Synthetic runs through the real class
(`reconciler_cases.py`) show the failure modes:

| case | input | result |
|---|---|---|
| window slides past a voice (buffer purge or force-finalize) | run A: speaker A at 0 to 10 s; run B: same A at 12 to 22 s | second run is **spk_1**: the same voice gets a new label |
| speaker leaves and returns beyond the window | A then B, B alone for 30 s, A returns | A returns as **spk_2** |
| segmentation splits one speaker, then merges | A; A+B; A; A+B | labels ping-pong and a third label is minted |
| 0.4 s backchannel ("yeah") inside another speaker's turn | A plus a short B, then A plus a short C | **every** short raw label mints a new session label (spk_1, spk_2) |
| raw labels permuted with unequal durations | A/B then B/A | handled correctly by the vote |
| two previous speakers merged into one raw label | A,B; A only; A,B | the next split mints spk_2 for B |

Consequences seen in the baseline: 5 to 7 labels minted for 4 speakers,
3 to 5 label flips per minute, 4 of 16 onsets never labeled on ES2004a. The
fork_on run minted 6 labels in 180 s and showed only spk_2 and spk_5 to the
viewer. The known-limitations section of `docs/speaker_diarization.md`
already names the cause; the fix is speaker-embedding memory:

- Keep a per-session bank of centroid embeddings per session label (pyannote
  exposes the embeddings it computed, so no extra inference), match each
  run's clusters to the bank by cosine distance with a threshold, and mint
  only when no centroid is close. Also require a minimum cumulative duration
  (for example 1.5 s) before a new label is minted; shorter speech inherits
  the nearest existing label or `null`. 3 to 5 days including a benchmark
  that reports labels minted versus reference speakers.

### 3.3 First seconds of a session. P2.

The first tick sees 5 s of audio and the profile shows pyannote handles a 5 s
input in 0.6 s with one speaker, so there is no crash, but with fewer than
10 s of audio the segmentation window is padded and the first labels are the
least reliable; with the two-pass LocalAgree nothing is shown before the
second tick anyway. Combined with section 2 the first label currently
arrives after 85 s. Once the tick fits the period, expect first labels after
about 10 s; document that and show "Speaker" without a number until a label
has survived two ticks. Half a day.

### 3.4 Noise and silence creating false speakers. P2.

Pyannote's segmentation has `min_duration_off: 0.0` in the shipped config and
no minimum turn length is applied in the fork, so far-field noise bursts
become sub-second turns that mint labels (section 3.2). ES2004a's false
alarm term is 0.23 first-seen. Fix: ignore raw segments shorter than 0.5 s
for minting (not for attribution) and feed Silero's speech ranges into
attribution so words outside speech never get a label. Half a day.

### 3.5 Under-counting speakers (TS3003a). P1.

Offline pyannote found one speaker in TS3003a with confusion 0.01 and missed
0.13, so the VBx clustering (`threshold: 0.6`) merged four far-field voices
into one and the remaining speakers' speech is scored as missed. This is a
model-and-microphone limitation, not a code bug: the AMI single distant
microphone is much harder than a classroom lapel or kiosk microphone, and the
fork passes no `min_speakers`. Options: expose and document
`diarization_min_speakers` for known-panel sessions; test the clustering
threshold at 0.5 and 0.55 on the three files; add a headset or close-talk
recording to the benchmark so the number customers will see is measured, not
the worst case. 1 day.

## 4. Streaming behavior

- **Time until a word gets its label.** Labels ride on the word
  (`_word_segment`, `whisper_streaming_job.py:623-637`), so label latency
  equals caption latency: 84.8 s mean in the fork_on run (65 labeled words
  in 180 s), 18 to 24 s plus the tick cost in the replay benchmark. P0, same
  fix as section 2.
- **How often a label changes after being shown.** Committed words freeze
  their label at commit (`local_agree.py:228-233` commits the latest pass's
  segment, including its speaker); `final` words never change afterwards.
  In the fork_on run no word changed its label between first display and
  finalization (0 of 65). The in-progress tail re-labels every tick, but the
  webapps never render in-progress speakers
  (`transcription-content-slice.ts:159-164` joins text only), so the user
  does not see those flips. What the user does see is a wrong label frozen
  forever, because nothing retroactively relabels (documented limitation).
- **What the user sees meanwhile.** Unlabeled words are absorbed into the
  previous speaker's run with no cue (`speaker-runs-text.tsx:36-42`), so a
  `null` word reads as if the previous speaker kept talking. P2: render an
  explicit "Speaker ?" or no label for null runs. 2 hours.
- **Reconnects reset labels** to `spk_0` with no signal to the viewer, see
  section 5.3.

## 5. Robustness

### 5.1 Model load failure, missing HuggingFace access. P1.

`PyannoteDiarizationContext.create()` (`pyannote_diarization_context.py:120-144`)
raises if the token env var is empty (clear message) and otherwise calls
`Pipeline.from_pretrained` with no try/except, so gated-repo, network and
bad-model errors propagate raw. The worker reports the failure and exits
(`worker_process.py:196-203`, `514-520`), the manager raises
(`worker_process_manager.py:528-551`), the registry and the uvicorn factory
raise, and the service never binds: no readiness, no health, no metrics.
That is fail-fast, which is right, but:

- Whisper and Silero are loaded first (config order), so a bad token wastes
  the Whisper load before dying.
- `transcription_service/provider_config.template.json` lists the pyannote
  context on worker 0 with `diarization_detector: false`, so anyone copying
  the template needs the token and the extra just to boot with diarization
  off.
- The model is downloaded at worker start, not "at the first diarization
  run" as `docs/speaker_diarization.md` says; the container needs outbound
  HTTPS to huggingface.co at startup unless the model is baked in.

Fix: guard `from_pretrained` returning `None` (older pyannote behavior),
wrap HF errors with the model URL and env var name, remove the context from
the service template or comment it, and document baking the model with
`HF_HUB_OFFLINE=1`. Half a day.

### 5.2 Worker crash or restart mid-session. P1 (upstream).

There is no worker restart. A worker that dies after init is invisible to
the result poll loop (`worker_process_manager.py:424-434`, `570-602`),
registered jobs never complete and never error, the client socket stays
open and silent, readiness goes 503, yet `_assign_process`
(`worker_pool.py:272-315`) can still route new sessions to the dead worker
because it scores utilization without checking `alive`. The
`SpeakerReconciler` lives inside the pickled job object in the worker
(`whisper_streaming_job.py:177`), so its state is lost with the worker;
labels restart at `spk_0`. Fix: respawn on death and filter dead workers in
`_assign_process` (upstream), and keep a label epoch on the Python session
so a restart can at least tell the client that labels were reset. 2 to 3
days upstream, half a day fork side.

### 5.3 Reconnects. P1.

Every WebSocket gets a new job and a new `SpeakerReconciler`
(`whisper_streaming_provider.py:152-169`). node-server reconnects to the
Python service automatically on any close code other than 1000 and 1001
(`websocket-client.ts:123-132`, `create-transcription-service-client.ts:20-73`),
so after any upstream blip the same people reappear as `Speaker 1`,
`Speaker 2` again, and the store happily groups them as new runs
(`speaker-runs.ts:22-28`). Fix: emit a `speakers_reset` marker (or a label
epoch) on the transcript message after reconnect and show a divider in the
UI; proper continuity needs the embedding bank from section 3.2 to be keyed
by session uid in node-server or persisted in the Python session cache.
1 day for the marker.

### 5.4 Several concurrent sessions sharing the worker pool. P1.

Routing requires one worker that owns all three contexts
(`worker_pool.py:244-260`, `294-314`). The template pins pyannote to worker
0, so with diarization on every session lands on worker 0 and extra workers
idle. Jobs on one worker execute serially, so the worker's throughput is
`job_period / tick_cost` sessions: with a 25 s tick that is zero sessions at
the 5 s period, with a 3.6 s tick one session per worker. No capacity
estimator is wired in deployment (`create_webserver.py:101-103`,
`capacity_estimator=None`), so overload never refuses with 1013; it just
drops periods (`asr_dropped_periods_total`) and marks the provider DEGRADED.
If no single worker owns all tags, the first audio chunk raises
`RuntimeError` and the socket closes 1011, which node-server retries
forever. Fix: see section 7.3 for the startup check; for capacity, document
"one diarized session per worker" until the tick fits, and give the pyannote
context `worker_ids` for every worker in the template. Half a day.

### 5.5 Memory growth over a two-hour session. P2.

Bounded: `SpeakerReconciler._previous` is replaced each run,
`speakers` lists are per result, LocalAgree queues are bounded by
`local_agree_dim` and force-finalize, the incremental VAD purges its
probabilities. Unbounded (upstream): `_chunk_ledger` is pruned only inside
`_extract_chunk_ids_for_time`, which returns early when there is no
transcript end time (`whisper_streaming_job.py:652-653`), so during any long
stretch without words (muted microphone, break) it grows one dict per
0.5 s chunk until the next transcript. Two hours of silence is 14,400
entries, small in bytes but a real leak. pyannote's own per-call allocations
were not seen to accumulate in a 180 s run (RSS flat at 1.9 GB after the
first tick); a two-hour soak was not run and should be before release. Fix
the prune ordering upstream (1 hour) and add a two-hour soak to the
benchmark harness (half a day).

### 5.6 Exceptions swallowed without a metric. P2.

| location | failure | counted | logged |
|---|---|---|---|
| `whisper_streaming_job.py:327-341` | pyannote raised | `diarization_failed` | warning every tick, every session, no rate limit |
| `worker_process.py:50-55` | `drain_counters` raised | no | error, counters for that tick lost |
| `worker_process.py:428-444` | `update_config` raised | counters not drained | yes |
| `worker_process_manager.py:586-602` | poll loop ended on `OSError`/`EOFError`/`RuntimeError` | no | **not logged** |
| `worker_process_manager.py:666-670` | metrics observer raised | no | error, tick metrics lost |
| `transcription_provider_registry.py:370-382` | `describe_health` raised | no | not logged, reported as DOWN |

Fix: rate-limit the diarization warning (first, then every 60 s) and add a
counter plus a log line to the silent poll-loop exit. 2 hours.

## 6. Performance on CPU

Per diarized session on this machine: about one core continuously (worker
CPU mean 104 percent, max 168), 1.9 GB RSS per worker holding Whisper base,
Silero and pyannote (0.9 GB without pyannote), 25.5 s mean tick against a
5 s period. The session cannot keep up, so the honest capacity today is
**zero diarized sessions per worker** at the 5 s period on CPU, and one
session only if the period is raised past the tick.

Projection once diarization is cut to the newest 10 s (0.6 s) and Whisper
runs one call per window (2.6 s, or 1.9 s single-threaded):

| configuration | tick cost | sessions per worker (5 s period) | workers that fit 16 GB (2 GB each, OS and Postgres aside) | sessions per machine |
|---|---|---|---|---|
| today | 25 s | 0 | 6 | 0 |
| 10 s diarization window, VAD-split Whisper | about 12 s | 0 | 6 | 0 |
| 10 s window, single-call Whisper, 4 threads | about 3.5 s | 1 | 4 to 6 | 4 to 6 |
| 10 s window, single-call Whisper, 1 thread per worker | about 2.6 s | 1 to 2 | 4 to 6 | 6 to 10 |
| diarization off, upstream defaults | 3.6 s mean | 1 | 8 | 8 |

These are estimates from the component timings; they need the Linux
measurement from section 11 step 2 before being promised to anyone. The
baseline's `peak_rss_mb: 2161` matches the 1.9 GB worker RSS.

## 7. Operations

### 7.1 Telemetry through upstream's monitoring. P1.

The five counters (`diarization_runs`, `diarization_seconds`,
`diarization_failed`, `reconciler_seconds`, `diarization_labels_minted`;
`job_counters.py:95-117`, `metrics_registry.py:217-238`, `286-300`) are
accumulated but **never exported**: `metrics_controller.py` builds
`/metrics/status` from a fixed field list and was not touched by the fork
(`git diff staging...HEAD --stat` has no entry for it), the monitoring
sidecar polls a fixed TypeBox schema
(`apps/monitoring-sidecar/.../transcription-metrics.schema.ts`), and
`grep -rn diariz apps/monitoring-sidecar deployment` returns nothing, so
Prometheus, the Grafana fleet dashboard and the alert rules cannot see
diarization at all. There is no per-tick diarization histogram, only totals.
Upstream's `asr_dropped_periods_total` and `asr_execution_ms` histograms do
exist and already include diarization time because it runs inside
`process_batch`. Fix: add the five counters to `/metrics/status`, the
sidecar schema, poller and registry, a Grafana panel for
`diarization_seconds_total / diarization_runs_total` against
`job_period_ms`, and an alert when that ratio exceeds 0.5. 1 day.

### 7.2 Lint coverage after the display-ui scope was narrowed. P2.

Per package `lint` / `format` scripts and the vitest includes:

| fork file | eslint | prettier check | vitest |
|---|---|---|---|
| `transcription-content-store/src/speaker-runs.ts`, slice, index | yes (`./src ./tests`) | yes | yes |
| `transcription-content-store/tests/unit/*.test.ts` | yes | yes | yes |
| `transcription-display-ui/src/components/speaker-runs-text.tsx`, container, `utils/speaker-appearance.ts`, index | yes (`./src`) | yes | n/a |
| `transcription-display-ui/tests/unit/speaker-appearance.test.ts` | **no** | **no** | yes |
| schema package edits | yes | yes | n/a |

`docs/upstream_sync_2026-10.md` says the display-ui tests are formatted
"through the package's prettier configs"; nothing under `tests/` is checked.
`npx eslint` and `npx prettier --check` on that one file both pass today, so
`"lint": "eslint ./src ./tests/unit"` and the same for `format` in
`libs/ui/transcription-display-ui/package.json` would cover it without
pulling in upstream's non-conforming test helpers. 1 hour, and fix the sync
doc sentence.

### 7.3 Config validation and fail-fast. P1.

`whisper_streaming_config.py:37-40` has no validator for
`diarization_min_speakers` / `max_speakers` (positive, min at most max) and
no check that `diarization_context_tag` resolves to a context. A missing or
wrong tag fails at the first audio chunk with `KeyError` or `RuntimeError`
from `worker_pool.py:294-314`, closes the socket 1011, and node-server
reconnects forever; `/providers/health` says DOWN but readiness stays 200.
The deprecated upstream branch (`Deprecated-issue-64-speaker-diarization`,
commit 5283eaa) asserted in the provider constructor that
`get_context_ids_by_tag(diarization_context_tag)` is non-empty and that the
context is a `PyannoteDiarizationContext`, logging and re-raising. That
helper no longer exists on `WorkerPool`, but `load_for_tags` does. Fix: in
`WhisperStreamingProvider.__init__`, `if not
worker_pool.load_for_tags(self.job_context_tags): raise ValueError(...)`
naming the tag and the configured contexts, plus a pydantic
`model_validator` for the speaker bounds. 2 hours.

### 7.4 Clear error messages. P2.

Runtime diarization failures are invisible to viewers (words simply lose
their labels, `local_agree.py:58-61` drops the array) and noisy for
operators (one warning per tick per session). Startup failures for gated or
unaccepted models are raw HuggingFace exceptions. Fix: wrap with "accept the
terms at https://huggingface.co/pyannote/speaker-diarization-community-1 and
set HUGGINGFACE_ACCESS_TOKEN", and rate-limit the runtime warning. 2 hours.

### 7.5 Docs for deployers. P1.

`deployment/` has no mention of diarization, pyannote, the token or the
extra (`compose.yml`, `UPGRADING.md`, both provider templates). The
production images install only `faster-whisper` and `silero-vad-cpu`
(`Dockerfile_CPU:21-28`, `Dockerfile_CUDA:52-59`), so enabling the context
in production fails at worker init with `ModuleNotFoundError`. Fix: a build
arg that adds `--extra pyannote-diarization`, pass the token through
`compose.yml`, an `UPGRADING.md` entry with the CC-BY attribution note, and
a commented context block in `deployment/provider_config.template.json`.
1 day.

## 8. UX and accessibility (WCAG 2.1 AA)

### 8.1 Label wording. P2.

`spk_N` renders as `Speaker N+1:` in bold (`speaker-appearance.ts:145-151`,
`speaker-runs-text.tsx:87-98`); any other label renders verbatim. English
only (no i18n layer exists in the repo), no renaming, no "unknown speaker"
state. Suggested: keep the wording, add a rename map in the content store
with a small menu, and an explicit unattributed state. 1 to 2 days for
rename, 2 hours for the unattributed state.

### 8.2 Color contrast. P1.

`getSpeakerColor` (`speaker-appearance.ts:115-139`) nudges an 8-color
Okabe-Ito palette toward white when the background luminance is below 0.5,
otherwise toward black, until 4.5:1 or 20 steps. For backgrounds with
luminance roughly between 0.18 and 0.5 (for example `#808080`) white can
never reach 4.5:1 (its maximum is 3.95:1) while black would, so the function
returns a failing color. The unit test checks only pure white and pure black
backgrounds. Upstream already has the right heuristic and helpers in
`libs/ui/theme-customization-ui/src/utils/color-contrast.ts:44-68`
(`readableTextColor` compares both targets). On dark presets every speaker
converges toward near-white, so colors stop distinguishing speakers, and the
label ignores the user's chosen transcription color entirely. Use of color
alone is not relied on (bold text label plus line break), which satisfies
SC 1.4.1. Fix: choose the target by the larger of the two contrasts, reuse
upstream's helpers, add preset-background tests, and fall back to the user's
transcription color when it has more contrast than the palette color. Half
a day.

### 8.3 More speakers than colors. P2.

Palette index is `N mod 8` (`speaker-appearance.ts:97-101`); speaker 9 shares
speaker 1's color. With the current minting behavior (6 labels in 3 minutes)
this happens in ordinary sessions. Fix: once minting is fixed this is rare;
until then, reuse colors only after the bold label and never adjacent. 2
hours.

### 8.4 Screen readers. P1 (inherited from upstream, amplified by speakers).

The container puts committed sections inside the `role="log"`
`aria-live="polite"` region and everything else, including finalized
sequences of the active section and the interim text, inside an
`aria-hidden="true"` block (`transcription-display-container.tsx:233-294`).
Committed sections are created only by `commitParagraphBreak`, which is
dispatched solely by the standalone webapp's WebSpeech provider middleware
(`apps/standalone-webapp/.../provider-service-middleware.ts:105,144,155`);
the client and kiosk webapps never dispatch it (no other call sites in
`apps/` or `libs/`). So in the networked apps no caption text and no speaker
label is ever announced to a screen reader, and braille users get nothing.
When sections are committed, the label is plain inline text before the
words, so "Speaker 2: ..." is read in order, which is the right shape. Fix:
dispatch a paragraph commit in the client and kiosk middleware on speaker
change, on a silence gap, or every N finalized sequences; the speaker change
is the natural boundary now that it exists. 1 day including an axe test with
speakered sequences (the package already has `tests/a11y.ts`).

### 8.5 Font size and user theme. P2.

Labels inherit the user's font size and spacing (`speaker-runs-text.tsx:89-97`
sets only color and weight), so they scale. Forced-colors mode will strip
the colors and leave the bold labels, which is acceptable.

## 9. Licensing and privacy

| component | license | notes |
|---|---|---|
| `pyannote/speaker-diarization-community-1` (segmentation, embedding, PLDA bundle) | **CC-BY-4.0** per the model page | gated: the deployer's HuggingFace account must accept terms and supply contact details, with consent to occasional email from pyannote; attribution is required in product documentation or an about screen; commercial use allowed |
| pyannote.audio 4.0.7 | MIT | |
| faster-whisper 1.2.1, CTranslate2 4.7.1, `Systran/faster-whisper-base` weights | MIT | |
| Silero VAD | MIT (`~/.cache/torch/hub/snakers4_silero-vad_master/LICENSE`) | |
| torch 2.13 | BSD-3 | |

The fork's README attribution covers upstream ScribeAR but not the CC-BY
notice for the model; add it to the deployer docs and the kiosk or client
about page (P1 for a customer release, 1 hour).

Privacy: no audio or embeddings are written to disk by the fork. Audio lives
only in the worker's in-memory circular buffer (two times
`max_buffer_len_sec`, purged as it finalizes); the reconciler keeps only the
previous run's `(start, end, label)` segments for that session, in the
worker process, and they die with the session. No speaker embeddings are
retained at all today (which is also why continuity is poor). The planned
embedding bank (section 3.2) would hold voiceprints in memory for the
session's lifetime; it should stay in memory, be dropped on session close,
and be documented as such, because voiceprints are biometric data under
GDPR and Illinois BIPA. Other data flows: the HuggingFace token sits in the
container environment; the first start downloads the model from
huggingface.co (set `HF_HUB_DISABLE_TELEMETRY=1`, bake the model and run with
`HF_HUB_OFFLINE=1` in production); speaker labels travel to every viewer in
the room and into node-server's transcript bus but are not persisted by the
session manager or the export (exports drop speakers, section 8). P1 to
write this down for customers, 2 hours.

## 10. Upstream branches reviewed (read-only)

| branch | state vs staging | what it contains | reuse |
|---|---|---|---|
| `latency-metrics` | 13 ahead, 514 behind, never merged; predates the worker-pool and session-manager rearchitecture | a `LatencyTracker` mark/delta class (`latency_track/latency_tracker.py`) with per-stage marks (audio received, VAD start and end, first transcript, Whisper first token), an end-to-end tracker through the old node-server, and a Silero config exposure | the idea only: per-stage marks inside one tick (VAD, diarization, Whisper, reconciler) exported as counters, which the fork already half has; the code will not apply |
| `latency-metrics-v2` | merged as PR #124 (plus PR #131 keeping heavy ML deps out of unit-test imports) | SAFP frames with chunk ids, `in_progress_chunk_ids` / `final_chunk_ids`, node-server latency percentiles (`latency-window.ts`), the client latency badge (`#metrics`, key `m`) | already in the fork's base; **use the chunk-id method as the Phase 2 caption-latency metric** so numbers match node-server's badge; PR #131's import-isolation pattern should be applied to the pyannote imports in unit tests |
| `web-gpu-integration` | fully merged, 0 ahead | monitoring per-device alert thresholds with automatic CPU/GPU selection, GPU made opt-in in compose | the per-device alert thresholds (`monitoring`) are where a diarization tick-ratio alert belongs |
| `apple-silicon-prep` | fully merged as PR #191 (with `feat/apple-silicon-support` PR #172) | multi-arch image publishing on native arm64 runners, compose GPU reservation moved behind an override, template tweaks | nothing diarization-specific; confirms that the arm64 CPU image is the supported Mac path and that there is no Metal or MPS acceleration for CTranslate2 (`explore/apple-silicon:APPLE-SILICON-FINDINGS.md` section 3), which closes the "Apple Silicon acceleration" question in section 1 |
| `worker-pool-update` | 4 ahead, 489 behind, superseded by PR #111 which is in the base | the original "initialize contexts immediately" change | nothing new |
| `Deprecated-issue-64-speaker-diarization` | 1 untested commit | pyannote context wired into the old provider, constructor assertions that the diarization tag resolves and is a pyannote context | the fail-fast check (section 7.3) |
| `cpu-findings.md` on staging and `tools/asr-load` | in base | OpenBLAS spin-wait diagnosis, `cpu_threads` defaults, cores-per-session load tool | run `npm run asr:load` with diarization on in the Docker stack on Linux as the Phase 2 acceptance test; the OpenBLAS issue does not apply on macOS (Accelerate) but does in the Linux image |

## 11. Prioritized fix plan

Phase 2a, make the tick fit (blocks everything else):

1. **Shrink the diarization window to the newest 10 to 15 s** with the
   reconciler keeping continuity, and rerun
   `make benchmark_diarization_baseline` (expect tick cost under 2 s). P0,
   1 to 2 days.
2. **Measure on Linux CPU** (the real target): the same probe and
   `npm run asr:load` in the CPU Docker image on a server, with
   `num_threads` for pyannote configurable and the Silero thread override
   accounted for. Decide thread counts from that run, not from this laptop.
   P0, 1 day.
3. **Fix the upstream caption path** that diarization inherits: one Whisper
   call per window with Silero ranges used as a mask, and a sane
   `silence_threshold` default; raise both upstream. P1, 2 days.
4. **Export the diarization counters** through `/metrics/status`, the
   sidecar and Grafana, with an alert on diarization seconds per tick over
   half the period. P1, 1 day.

Phase 2b, make labels trustworthy:

5. **Embedding memory in the reconciler** with a minimum duration before
   minting, and overlap-aware diarization output. P1, 1 week including
   benchmark metrics for labels minted and onsets never labeled.
6. **Reconnect and restart semantics**: label epoch marker on the wire, UI
   divider, and the upstream dead-worker routing fix. P1, 2 days.
7. **Speaker-count bounds and threshold tuning** with a close-microphone
   recording added to the benchmark. P1, 1 day.

Phase 2c, ship-readiness:

8. **Fail-fast config checks** (tag resolution, speaker bounds) and clearer
   HF error messages, rate-limited runtime warnings. P1, half a day.
9. **Deployment**: pyannote extra behind a build arg in both Dockerfiles,
   token through compose, `UPGRADING.md` entry, commented context in the
   deployment template, CC-BY attribution, privacy note, offline model
   baking. P1, 1 day.
10. **Accessibility**: paragraph commits in client and kiosk so the live
    region announces turns; contrast target fix with preset tests; explicit
    unattributed state; color reuse rule. P1, 2 days.
11. **Export and translation** carry speakers (export 4 hours, translation
    1 to 2 days); node-server and display tests for `speakers`; lint scope
    for the display-ui test file. P2, 2 to 3 days.
12. **Two-hour soak** in the benchmark harness and the `_chunk_ledger` prune
    fix upstream. P2, 1 day.

Total: roughly four to five engineering weeks, of which the first two items
decide whether CPU-only diarization is viable for customers at all.
