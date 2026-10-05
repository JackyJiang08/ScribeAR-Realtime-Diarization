# Upstream issue drafts for scribear/scribear

Drafts for issues against the upstream repository, written from measurements
taken on this fork during the diarization work (Phase 2 audit,
`docs/diarization_production_audit.md`, 2026-10-02, and the Phase 2a runs,
2026-10-03). Nothing here changes upstream's Whisper or VAD path in the fork;
the fork's benchmarks use upstream's shipped configuration unchanged so that
these numbers describe upstream, not the fork.

Everything was measured with upstream's `whisper-streaming` provider,
faster-whisper `base` on CPU, 5 s `job_period_ms`, 30 s `max_buffer_len_sec`,
`local_agree_dim` 2, English, streaming AMI meeting ES2004a (single distant
microphone, 16 kHz mono) at real time in 0.5 s SAFP frames straight at
`/transcription_stream/whisper`. Two machines:

- **Laptop**: Apple M4 (4 performance + 6 efficiency cores), 16 GB, macOS
  15.7, Python 3.12.11, faster-whisper 1.2.1, CTranslate2 4.7.1, torch 2.13.
- **Reference container**: upstream's `Dockerfile_CPU` image, run under
  Docker Desktop on that same Mac with `--cpus 4 --memory 8g`
  (`linux-cpu-4c8g`; the Docker Desktop VM has 7.65 GB, so the 8 GB limit is
  nominal). This is Linux and upstream's own image, thread settings included
  (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `cpu_threads` 4), but it is
  a VM on a laptop, not a server. Where a number comes from this container
  it is labelled as such so you can judge whether it reproduces on real
  hardware.

---

## Issue 1: `vad_detector: true` makes one Whisper call per Silero range, 3 to 5x the cost of one call per window

**Where.** `transcription_service/src/transcription_providers/whisper_streaming_provider/whisper_streaming_job.py`,
`_transcribe_audio`: with `vad_detector` on, `_detect_speech_ranges` returns
the Silero speech ranges of the window and the loop below calls
`whisper.transcribe(...)` once per range.

**What happens.** faster-whisper pads every call to a 30 s encoder window,
so a 0.4 s range costs about as much as a 10 s one, and the per-call
temperature fallback and silence-skip logic occasionally make a sub-second
range cost several seconds on its own. The window is re-transcribed every
period, so the multiplier applies to every tick.

**Measurements** (laptop, `_transcribe_audio` replayed outside the service on
eight 30 s windows of ES2004a with upstream's own `SileroVadContext` stream and
`FasterWhisperContext`, the previous window's words as `initial_prompt`, three
repeats after warm-up):

| window | Silero ranges | per-range calls, total | words | one call per window |
|---|---|---|---|---|
| 0 to 30 s | 8 | 11.5 s (one 0.4 s range: 7.1 s) | 18 | 3.3 s |
| 10 to 40 s | 8 | 8.4 s | 30 | 9.4 s (see issue 2) |
| 20 to 50 s | 8 | 12.8 s (one 0.3 s range: 5.8 s) | 13 | 5.3 s |
| 90 to 120 s | 14 | 13.2 s | 64 | 2.8 s |
| 100 to 130 s | 13 | 16.9 s (one range 7.1 s, one 3.8 s) | 44 | 3.1 s |
| 115 to 145 s | 15 | 14.4 s | 50 | 2.9 s |
| 120 to 150 s | 16 | 10.5 s | 60 | 3.2 s |
| 145 to 175 s | 13 | 7.3 s | 77 | 3.2 s |

End to end on the laptop (plain upstream `staging`, 180 s of ES2004a, 36
periods of 5 s):

| config | words first shown, mean / p95 | newest audio to in-progress caption (chunk-id), mean / p50 | periods dropped | tick exec mean / max |
|---|---|---|---|---|
| `vad_detector: true` | 30.2 s / 50.8 s | 18.1 s / 20.3 s | 17 of 36 | 6.7 s / 32.5 s |
| shipped default (no VAD) | 18.8 s / 32.3 s | 7.0 s / 3.6 s | 10 of 39 | 3.6 s / 38.6 s |

Thread caps (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, the image's
setting) and a 1 s period change nothing; the cost is in the calls.

**Suggestion.** Transcribe the whole window once and use the Silero ranges to
mask or trim silence (zero out, or cut the window to the span from the first
range start to the last range end), or merge ranges separated by less than
about 1 s before calling Whisper. Either keeps VAD's benefit (no hallucinated
text in silence) at the cost of one call per tick. Worth doing even though the
shipped template does not enable `vad_detector`: anyone who turns it on today
gets 3 to 5x the tick cost with no warning.

---

## Issue 2: `silence_threshold` defaults to 0.01 s, which makes faster-whisper re-decode around every 10 ms pause

**Where.** `whisper_streaming_config.py`: `silence_threshold: float = 0.01`,
passed straight to `whisper.transcribe(hallucination_silence_threshold=...)`
in `whisper_streaming_job.py`.

**What happens.** faster-whisper's `hallucination_silence_threshold` is meant
to skip silent stretches "longer than this many seconds" when a hallucination
is detected; at 0.01 s it triggers on effectively every pause, and the model
re-seeks and re-decodes around each one. On a sparse window that turns one
2.6 s call into 5 to 9 s. This is in the shipped default, so it affects every
deployment, with or without VAD.

**Measurements** (laptop, same replay as issue 1, one call per 30 s window):

| window | with `hallucination_silence_threshold=0.01` | without the argument |
|---|---|---|
| 0 to 30 s | 3.3 s | 2.0 s |
| 10 to 40 s | **9.4 s** | 2.6 s |
| 20 to 50 s | 5.3 s | 2.7 s |
| 90 to 120 s | 2.8 s | 2.8 s |
| 100 to 130 s | 3.1 s | 3.0 s |
| 115 to 145 s | 2.9 s | 3.0 s |
| 120 to 150 s | 3.2 s | 3.2 s |
| 145 to 175 s | 3.2 s | 3.1 s |

Dense windows are unaffected; sparse ones (the start of a session, a pause,
a quiet room) pay 2 to 3.5x. Isolated `WhisperModel("base",
cpu_threads=4)` on one 30 s window costs 2.66 s on this laptop, so the
re-decoding is the whole difference.

**Suggestion.** Default `silence_threshold` to something near 2 s, or to
`None` (faster-whisper's own default disables the behaviour), and document
it as a hallucination guard rather than a performance setting.

---

## Issue 3 (observation, please judge on real hardware): the shipped CPU config drops 11 of 36 periods with a single session in a 4-CPU container

**What was measured.** Upstream's `Dockerfile_CPU` image, shipped
`deployment/provider_config.template.json` whisper provider (faster-whisper
`base`, no VAD, `cpu_threads` 4 from the context default, 5 s period, 30 s
buffer), one session streaming ES2004a at real time for 180 s, nothing else
running in the container. **Measured under Docker Desktop on a Mac with
`--cpus 4 --memory 8g`**, so a VM on a laptop, not a server; the numbers
are reported so you can judge whether they reproduce on the hardware you
deploy on.

| metric (reference container, diarization off, i.e. plain upstream) | value |
|---|---|
| periods dropped (`asr_dropped_periods_total`), 36 scheduled | **11** |
| audio dropped because the buffer was full (`audio_dropped_buffer_full_seconds_total`) | 4.3 s |
| audio force-finalized by buffer overflow | 139.9 s |
| newest audio to in-progress caption (chunk-id), p50 / p95 | 6.0 s / 22.7 s |
| words first shown (word method), p50 / p95 | 17.2 s / 34.8 s |
| `asr_execution_ms` p95 / max | 31.5 s / 31.5 s |
| service RSS peak | 1.32 GB |

The same clip on the laptop natively (no container, 10 cores) drops 5 of 36
periods with p50 chunk-id latency 3.0 s. So with Whisper `base` alone, 4 CPUs
is already most of the budget: a 30 s window costs about 2.6 s in isolation
(laptop) but single live ticks reach 30 s under the container's 4-CPU
quota, and every period that elapses while a tick runs is dropped
(`worker_process.py`, `_advance_period`). If this reproduces on a real
4-vCPU server, the CPU template's single worker cannot serve one session
without losing captions, and the capacity estimator would start every
session in a degraded window.

Things that would help pin it down upstream:

- The isolated per-window cost on a real 4-vCPU Linux box
  (`WhisperModel("base", device="cpu", cpu_threads=4)` on a 30 s window).
- Whether `cpu_threads: 1` or 2 does better than 4 under a 4-CPU cgroup
  quota (on the laptop 1 thread was faster than 4: 1.86 s against 2.66 s,
  because the efficiency cores slow the pool down; a cgroup quota might
  behave similarly).
- Issues 1 and 2: both reduce the tick cost without touching the model.

The full reports with every counter are in this fork under
`transcription_service/benchmarks/diarization/baselines/linux-cpu-4c8g.json`
(`caption_latency.reference.off`) and `native-darwin-arm64.json`.

---

## Issue 4: in the client and kiosk webapps no caption text ever reaches the screen-reader live region

**Where.** `libs/ui/transcription-display-ui/src/components/transcription-display-container.tsx`
puts the **committed sections** inside the `role="log"` `aria-live="polite"`
region and everything else, the active section's finalized sequences and the
interim text, inside an `aria-hidden="true"` block (by design: interim text is
rewritten several times a second). Committed sections are created only by the
content store's `commitParagraphBreak` action, and the only code that
dispatches it is the standalone webapp's WebSpeech provider middleware
(`apps/standalone-webapp/src/features/transcription-providers/stores/provider-service-middleware.ts`,
on the provider's `commitParagraphBreak` event and on provider switches).
`apps/client-webapp` and `apps/kiosk-webapp` handle `transcript` events with
`handleTranscript` alone and never commit a paragraph.

**What happens.** In the two networked apps, `commitedSections` stays empty
for the whole session: every finalized caption lives in the active section,
which is hidden from assistive technology. A screen-reader or braille user
joining a room through the client webapp hears nothing at all, for the whole
session, while a sighted viewer sees every caption. axe does not catch it,
because the markup is correct; the region is simply never written to. (The
translated-captions panel has the same live-region shape and the same
problem is avoided there only because every translated segment is appended
directly into its region.)

**Confirmed how.** By code inspection on upstream `staging` (5f01575): a
search over `apps/` and `libs/` finds `commitParagraphBreak` dispatched only
in the standalone provider middleware, and the client and kiosk transcript
handlers only call `handleTranscript` / `applySpeakersUpdate`, which append
to the active section. Feeding `handleTranscript` events into the content
reducer leaves `commitedSections` empty (the fork's
`paragraph-commit-middleware.test.ts` asserts exactly that without the
middleware installed), and the container renders only `commitedSections`
inside the `role="log"` element. Not verified with a screen reader by the
fork's author; the markup makes the outcome deterministic.

**Suggestion.** Dispatch paragraph commits in the client and kiosk apps on
natural boundaries: a pause in finalized text (a few seconds without a new
final), a cap on sequences per paragraph (so an uninterrupted monologue is
still announced in pieces), and, where the provider labels speakers, a
change of speaker. The fork does this with a store middleware
(`createParagraphCommitMiddleware` in
`libs/store/transcription-content-store`, installed by the client and kiosk
stores; defaults: speaker change, 8 s idle, 6 sequences), and threads the
previous speaker across committed paragraphs so a speaker is named once per
turn and not again at every paragraph top. The fix is small and independent
of diarization; an idle-plus-length rule alone closes the gap for upstream.

