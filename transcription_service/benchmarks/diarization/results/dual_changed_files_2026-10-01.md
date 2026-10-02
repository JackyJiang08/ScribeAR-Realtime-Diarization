# Files changed on both feature/speaker-diarization and upstream/staging since fork point b3e08e5 (checked 2026-10-01)

| # | File | Mine | Upstream | Risk |
|---|------|------|----------|------|
| 1 | README.md | New fork README (feature table, setup, branch notes) | New monorepo README (layout, wiki links) | HIGH add/add |
| 2 | libs/schemas/.../transcription-stream.schema.ts | optional `speakers[]` on final/in_progress sequences | session_uid/room_uid, chunk ids, close code 1013 | LOW |
| 3 | libs/store/transcription-content-store/package.json | scripts cover ./tests, add test:unit | v0.3.0, dev import conditions, uuid 14 | LOW |
| 4 | libs/store/.../transcription-content-slice.ts | speaker-runs import, speakers?/runs? fields, runs computed in reducer | latency state + selectors inserted after same import line | MEDIUM |
| 5 | libs/ui/transcription-display-ui/package.json | scripts cover ./tests, add test:unit | v0.3.0, test:unit + testing-library/jsdom devDeps, MUI 9 | MEDIUM |
| 6 | libs/ui/.../transcription-display-container.tsx | SpeakerRunsText, memoized ActiveSequenceText, theme bg | role=log/aria-live, fillParentHeight, jumpToBottom, same JSX lines | HIGH |
| 7 | libs/ui/transcription-display-ui/src/index.ts | export speaker-runs-text, speaker-appearance | export jump-to-bottom-button, auto-scroll hooks | MEDIUM (keep both) |
| 8 | libs/ui/transcription-display-ui/vitest.config.ts | new, defineProject, node env | new, defineConfig, jsdom env, setupFiles | HIGH add/add (take upstream) |
| 9 | transcription_service/Makefile | install targets add --extra pyannote-diarization | tests lint disables duplicate-code | LOW |
| 10 | transcription_service/provider_config.template.json | pyannote-diarization context, diarization_* fields on whisper provider | context renamed whisper_context, model base, crisper_whisper + lumen_granite providers | MEDIUM-HIGH (fields may land in wrong block) |
| 11 | transcription_service/pyproject.toml | extra pyannote-diarization = pyannote.audio>=4,<5 | v0.3.0, torch 2.13/torchaudio 2.11, redis, dep bumps, silero conflicts block | MEDIUM |
| 12 | transcription_service/src/shared/config/config.py | PYANNOTE_DIARIZATION enum member | +177 lines env fields/validators, enum untouched | LOW |
| 13 | .../whisper_streaming_config.py | four diarization_* fields after silence_threshold | guard thresholds + force_finalize inserted at same spot | MEDIUM (keep both) |
| 14 | .../whisper_streaming_job.py | JobContexts union, SpeakerReconciler, speaker detection/assignment, rewritten _transcribe | +489/-31: AudioChunkPayload, windows, guards, telemetry, same functions | HIGH (re-apply by hand) |
| 15 | .../whisper_streaming_provider.py | context_tags tuple incl. diarization tag passed to register_job | register_job moved to lazy _ensure_job; new context_tags property | HIGH (code moved) |
| 16 | .../transcription_stream_controller.py | speakers= on both TranscriptSequence builds | +171/-12 admission, metering, chunk ids nearby | LOW |
| 17 | .../server_messages.py | speakers field on TranscriptSequence | chunk id fields on TranscriptMessage | LOW |
| 18 | .../transcription_provider_registry.py | PYANNOTE_DIARIZATION case after SILERO_VAD | +433/-13 health/admission/capacity, _context_device_by_tag one line away | LOW-MEDIUM |
| 19 | transcription_service/uv.lock | pyannote stack + deps added | torch 2.13 cpu/cuda split, redis, many bumps | HIGH (regenerate with uv lock) |
