# Upstream sync, October 2026

How `feature/speaker-diarization` was brought up to date with upstream
`scribear/scribear` `staging` (5f01575, 2026-09-23) from the fork point
b3e08e5 ("feat: transcription stream healthchecks (#112)", May 2026).

The merge was done with `git merge --no-commit staging` and every conflicted
file was combined by hand. Rule applied throughout: upstream's structure wins,
the fork's diarization logic is re-implemented on top of it. No file was
resolved by taking one side wholesale. The merge is a faithful port, not a
redesign: the tick loop, the 30 s rolling buffer and the reconciler are
unchanged, so a change in the benchmark after this merge is attributable to
the merge alone.

Safety net: branch `backup/speaker-diarization-pre-sync-2026-10-01` and tag
`pre-upstream-sync-2026-10-01`, both at 4ce85ca, both on origin.

## Manually resolved files

### transcription_service/src/transcription_providers/whisper_streaming_provider/whisper_streaming_job.py

- **Upstream changed:** payloads are `AudioChunkPayload` (chunk id + bytes)
  instead of raw bytes; `_decode_audio` drops and counts oversize batches
  instead of raising; a bounded tail split (`force_finalize_len_sec`,
  `max_transcribe_len_sec`) so Whisper only sees the oldest W seconds;
  per-execution `JobCounterCollector` telemetry (buffer overflow, VAD
  no-speech, hallucination guards over `compression_ratio` / `avg_logprob` /
  `no_speech_prob` / temperature fallback, repeated segments); an incremental
  per-session VAD stream; an audio-level meter and VAD statistics published
  as audio stages; chunk-id correlation for latency; `_build_result`.
- **Fork had:** a `JobContexts` union that optionally carries the pyannote
  context as a third element; `SpeakerReconciler` set up in `__init__`;
  `_detect_speaker_ranges` diarizing the whole buffer each tick and shifting
  results to session time; `_assign_speaker` (largest-overlap word
  attribution); `_transcribe_range` producing `TranscriptionSegment`s with a
  speaker; `_append_sequence` extending `speakers`.
- **Combined:** upstream's file is the base, verbatim. Diarization is added
  back as follows. The job keeps upstream's `_transcribe_audio(whisper,
  vad_context, log)` signature (upstream's tests call it directly); the
  pyannote context is taken from `contexts[2]` in `process_batch` and held on
  the instance for the pass, the same handoff upstream uses for the VAD
  stream. `_detect_speaker_ranges` diarizes exactly the `buffer_samples`
  slice Whisper transcribes (the front-anchored `max_transcribe_len_sec`
  window), so there is one audio source and one timeline; with the default
  config that window is the full 30 s buffer, as before. Word attribution
  happens where upstream appends each word to `transcription`, after the
  guard counters, so anything that does not reach the transcript never
  carries a label. Speakers still ride on `TranscriptionSegment` through
  LocalAgree, so a finalized sequence is emitted with the labels its words
  had when they were finalized, together with their final text. Diarization
  timing now goes through upstream's counters instead of log lines (see
  telemetry below). The diarization exception handler counts the failure as
  well as logging it.

### transcription_service/src/transcription_providers/whisper_streaming_provider/whisper_streaming_provider.py

- **Upstream changed:** `register_job` moved out of the session constructor
  into `_ensure_job()`, called on the first audio chunk, with
  `session_uid`/`room_uid`, provider key and admission undo; new
  `admission_worker_id`, `job_period_ms`, `context_tags` properties;
  `describe_health` checks that a live worker owns the whisper and silero
  contexts.
- **Fork had:** the diarization context tag appended to the registration
  tuple in the constructor when `diarization_detector` is on.
- **Combined:** upstream's file is the base. A new `job_context_tags`
  property builds the tuple (whisper, silero, plus the diarization tag when
  enabled) and is used both by `_ensure_job` and by `describe_health`, so
  registration and health cannot disagree. Because registration is lazy, a
  session that never sends audio never claims a worker that owns the pyannote
  context. Upstream's `context_tags` (device reporting) is left as upstream
  wrote it.

### libs/ui/transcription-display-ui/src/components/transcription-display-container.tsx

- **Upstream changed:** `role="log"` live-region attributes, `aria-hidden`
  on the interim text, `fillParentHeight`, `announceUpdates`,
  `idleReengageMs`, `overflowAnchor`/`overscrollBehavior`/focus-ring
  styles, the removal of the bottom sentinel box, and `jumpToBottom` wiring.
- **Fork had:** `SpeakerRunsText` inside `CommittedSections`, a memoized
  `ActiveSequenceText` per active sequence with previous-speaker tracking,
  and the theme background color passed down for readable labels.
- **Combined:** upstream's JSX, props and attributes verbatim; the fork's
  components slot into the same two places (committed sections, the
  `aria-hidden` interim `Typography`). Nothing upstream added was moved or
  reworded.

### transcription_service/pyproject.toml and uv.lock

- **Upstream changed:** version 0.3.0, torch 2.13.0 / torchaudio 2.11.0 in
  both silero extras, a `[tool.uv] conflicts` block separating `silero-vad`
  (CUDA, PyPI) from `silero-vad-cpu` (pytorch-cpu index), redis, many
  dependency bumps (pydantic, fastapi, starlette, uvicorn, websockets, pytest
  9, black 26, isort 8, pylint 4).
- **Fork had:** `pyannote-diarization = ["pyannote.audio>=4.0,<5.0"]` and a
  lockfile with the pyannote stack on torch 2.9.1.
- **Combined:** upstream's `pyproject.toml` with the pyannote extra re-added
  after the silero lines. It is not part of the conflict set, so it installs
  alongside either silero extra and shares whichever torch that extra
  resolved. The lockfile was taken from upstream and regenerated with
  `uv lock`, never hand-merged. Resolved pyannote.audio is 4.0.7, the same
  version the fork used before; it now runs on torch 2.13.0 (torchcodec
  0.17.0). An install without the extra is identical to upstream.

### transcription_service/provider_config.template.json

- **Upstream changed:** the whisper context tag was renamed
  `whisper_cpu_context` to `whisper_context`, the default model moved from
  `small` to `base`, and `crisper_whisper` and `lumen_granite` providers were
  added. Git auto-merged this file but put the fork's `diarization_*` fields
  under the new `crisper_whisper` provider.
- **Combined:** rebuilt from upstream's text. The `pyannote-diarization`
  context sits after the silero context; `diarization_detector: false` and
  `diarization_context_tag` sit under the `whisper` provider only. Upstream's
  default model is kept.

### README.md

- Both sides created the file independently. Upstream's README is the base.
  A fork-notes block (22 lines) sits above it with what the fork adds, the
  status and baseline numbers, how to enable it, and links to the diarization
  doc and this file. Everything else from the old fork README (testing
  checklist, layer table, UI check, acknowledgments) moved to
  `transcription_service/docs/speaker_diarization.md`.

### libs/ui/transcription-display-ui/vitest.config.ts

- Both sides created it. Upstream's version is used: `defineConfig`, an
  `include` of `./tests/**/*.test.{ts,tsx}`, the `jsdom` environment and
  `tests/setup.ts`. That include already picks up the fork's
  `tests/unit/speaker-appearance.test.ts`. The fork's
  `exclude: ['tests/integration/**']` was dropped because no such directory
  exists in this package.

### libs/ui/transcription-display-ui/package.json

- Upstream's block (test:unit, testing-library/jsdom/axe devDependencies, MUI
  9) is used as is. The fork had widened `format`/`lint` to `./src ./tests`;
  that was tried and dropped, because upstream's own test files (`setup.ts`,
  `render.tsx`, the control tests) do not pass upstream's eslint rules and
  upstream does not lint them. The fork's tests are still run by `test:unit`
  and formatted by prettier through the package's vitest and prettier
  configs. The file ends up identical to upstream.

### libs/ui/transcription-display-ui/src/index.ts

- Both export lists kept: upstream's jump-to-bottom button and auto-scroll
  hooks, the fork's `speaker-runs-text` and `speaker-appearance`.

### libs/store/transcription-content-store/src/transcription-content-slice.ts

- Both inserted after the same import line. The fork's `speaker-runs` import
  is followed by upstream's latency state, types and selectors. The fork's
  `speakers?`/`runs?` fields and the `runs` computation in the commit-section
  reducer auto-merged elsewhere in the file.

## Files that auto-merged and were checked

- `whisper_streaming_config.py`: both insertions after `silence_threshold`
  kept (fork's `diarization_*`, upstream's guard thresholds and bounded-tail
  fields with validator).
- `transcription_stream_controller.py`, `server_messages.py`,
  `transcription_sequence.py`, `local_agree.py`: fork's `speakers` fields
  intact next to upstream's chunk ids, admission and metering.
- `transcription_provider_registry.py`: the `PYANNOTE_DIARIZATION` case sits
  next to upstream's `_context_device_by_tag`. `PyannoteDiarizationContext`
  gained a `device` property so `/metrics/status` can report it, matching
  the whisper context.
- `config.py`: enum member intact. `Makefile`: upstream's
  `duplicate-code` lint flag plus the fork's install extras and the
  `benchmark_diarization_baseline` target.
- Schemas (`transcription-service-schema`, `node-server-schema`) and the
  store `package.json`: additive on both sides.
- `tests/manual/transcription_stream_file_client.py`: still matches
  upstream's auth/config handshake (`session_uid`/`room_uid` are optional).
- Nothing under `.github/workflows/` was touched.

## Telemetry added

`TranscriptionJobCounter` gained `diarization_runs`, `diarization_seconds`,
`diarization_failed`, `reconciler_seconds` and `diarization_labels_minted`,
reported per execution by the job and mapped to `*_total` counters in
`MetricsRegistry`. Seconds divided by runs is the per-pass diarization
latency; these replace the fork's earlier log-only timing.

## Diarization behavior changes caused by the merge

- None intended. Same 5 s tick, same 30 s buffer, same `SpeakerReconciler`.
- One consequence of upstream's bounded-tail split: diarization now covers
  the transcribe window (`max_transcribe_len_sec`) rather than the whole
  buffer. With the default config both are 30 s, so nothing changes; a
  deployment that sets a smaller window will diarize that window.
- Upstream's `_decode_audio` no longer raises on oversize batches, so the
  diarization timeline follows upstream's retained-samples accounting.
- Diarization errors are now counted as well as logged.

## Transcription stream client "starts/ends swap"

The deprecated upstream branch fixed a positional swap of `starts` and `ends`
in the old `transcription-stream-client.ts` emit calls. That client no
longer exists on staging: node-server forwards `msg.final` and
`msg.in_progress` as whole objects onto the transcript event bus and the
webapps read `starts`/`ends`/`speakers` by name from the schema-validated
message. The bug cannot occur in the current code, so no fix was needed.

## Post-merge verification (2026-10-02, merged tree before commit)

Python 3.12.11, venv synced to the regenerated lockfile (torch 2.13.0,
pyannote.audio 4.0.7). Node v25.8.0.

| Check | Result |
|---|---|
| `make format` (isort + black) | pass, 192 files |
| `make lint` (pylint src and tests) | 10.00/10 both |
| pytest unit | 540 passed, 2 failed (the same pre-existing timing-sensitive `worker_process_manager` tests) |
| `npm run build` (every workspace) | pass |
| transcription-service-schema, node-server-schema | build, lint, format pass |
| transcription-content-store | build, lint, format pass; 12 tests pass |
| transcription-display-ui | build, lint, format pass; 96 tests in 12 files pass |
| node-server | build, lint, format pass; 253 tests in 22 files pass |
| admin-webapp (untouched by the merge) | 559 tests pass with `NODE_OPTIONS=--no-experimental-webstorage`; under Node 25's built-in `localStorage` shim they all fail with `localStorage.setItem is not a function`, an environment issue unrelated to this merge |
| `npm run test:unit` (all 28 workspaces, same NODE_OPTIONS) | every workspace passes, 2,803 tests in total |

Pre-sync the Python suite had 162 tests; the merge brings upstream's new
tests, so the count rose to 542 with the same two known failures.

## Runtime verification after the merge (2026-10-02)

- **Benchmark rerun** (`make benchmark_diarization_baseline`, saved as
  `results/post_merge_baseline.json`): every DER / JER value and component,
  offline and streaming, label latency, flips and minted labels are
  identical to `pre_sync_baseline.json` to four decimals. Only timing moved
  (offline RTF 0.72 to 0.69, tick cost within 1 s), which is run-to-run
  noise. torch 2.13 therefore produced bit-identical pyannote output here.
- **Caption latency**, ES2004a first 180 s streamed at real time into the
  Python service (whisper base, Silero VAD, CPU, 0.5 s chunks), measured
  from the moment a chunk was sent to the first message showing its words:

  | | diarization off | diarization on |
  |---|---|---|
  | transcript messages in 180 s | 16 | 6 |
  | words first shown | 185 | 49 |
  | first-shown latency mean / p95 | 26.8 s / 39.5 s | 84.4 s / 105.5 s |
  | finalized latency mean / p95 | 48.0 s / 59.0 s | 80.2 s / 84.9 s |
  | job execution mean / max | 8.0 s / 32.8 s | 26.1 s / 74.4 s |

  Diarization on this CPU delays the captions themselves by roughly a
  minute and drops two thirds of the job periods. That is the Phase 2
  problem, unchanged by the merge.
- **End to end**: transcription service (diarization on), a stub session
  manager serving the session-config long-poll, and node-server built from
  this branch; a driver authenticated as kiosk and viewer with a locally
  signed session token, streamed 120 s of ES2004a as SAFP frames, and
  received transcripts with `speakers` through node-server. Every server
  message validated against the published node-server schema. No word
  changed its label between messages. The captured messages replayed
  through the transcription-content store into `TranscriptionDisplayContainer`
  rendered `Speaker 3:` in bold with the palette color `#007a59` inside the
  `role="log"` region.
- **Diarization off**: with the `pyannote` package made unimportable the
  service imports and all 540 unit tests pass; with `diarization_detector`
  false no message carries a `speakers` field.
- The manual file client was sending raw WAV chunks, which the merged
  controller drops as malformed frames; it now encodes SAFP frames
  (commit after the merge).

## Pre-sync verification (step 1, 2026-10-01, at 4ce85ca)

| Check | Result |
|---|---|
| isort --check | pass |
| black --check | pass |
| pylint src / tests | 10.00/10 both |
| pytest unit | 162 passed, 2 failed (pre-existing timing-sensitive `worker_process_manager` tests) |
| TS schema package build | pass |
| node-server build | pass |
| node-server unit tests | 49 passed, 5 files |

## Files changed on both sides since the fork point (pre-merge analysis)

| # | File | Fork | Upstream | Risk |
|---|------|------|----------|------|
| 1 | README.md | new fork README | new monorepo README | high, add/add |
| 2 | libs/schemas/.../transcription-stream.schema.ts | optional `speakers[]` | session/room uids, chunk ids, close code 1013 | low |
| 3 | libs/store/transcription-content-store/package.json | scripts cover ./tests, test:unit | v0.3.0, import conditions, uuid 14 | low |
| 4 | libs/store/.../transcription-content-slice.ts | speaker-runs import, speakers?/runs?, runs in reducer | latency state and selectors after same import | medium |
| 5 | libs/ui/transcription-display-ui/package.json | scripts cover ./tests, test:unit | v0.3.0, testing devDeps, MUI 9 | medium |
| 6 | libs/ui/.../transcription-display-container.tsx | SpeakerRunsText, memoized active text | a11y roles, fill-height, jumpToBottom | high |
| 7 | libs/ui/transcription-display-ui/src/index.ts | speaker exports | jump-to-bottom and auto-scroll exports | medium |
| 8 | libs/ui/transcription-display-ui/vitest.config.ts | new, node env | new, jsdom env + setup | high, add/add |
| 9 | transcription_service/Makefile | install extras | duplicate-code lint flag | low |
| 10 | transcription_service/provider_config.template.json | pyannote context, diarization fields | context rename, model base, two new providers | medium-high |
| 11 | transcription_service/pyproject.toml | pyannote extra | torch 2.13, conflicts block, bumps | medium |
| 12 | transcription_service/src/shared/config/config.py | enum member | +177 lines env fields | low |
| 13 | .../whisper_streaming_config.py | diarization fields | guard thresholds, bounded tail | medium |
| 14 | .../whisper_streaming_job.py | reconciler, speaker attribution | +489/-31 payloads, windows, telemetry | high |
| 15 | .../whisper_streaming_provider.py | context tag at registration | lazy `_ensure_job`, health | high |
| 16 | .../transcription_stream_controller.py | speakers= on sequences | admission, metering, chunk ids | low |
| 17 | .../server_messages.py | speakers field | chunk id fields | low |
| 18 | .../transcription_provider_registry.py | PYANNOTE case | health/admission/capacity, device map | low-medium |
| 19 | transcription_service/uv.lock | pyannote stack | torch 2.13 split, redis, bumps | high, regenerate |
