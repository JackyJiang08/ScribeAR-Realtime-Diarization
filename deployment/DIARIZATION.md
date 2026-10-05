# Speaker diarization in a deployment

How to run the transcription service with live speaker labels: what the
feature does, the images that carry it, every configuration key, what it
costs per session, how to verify a deployment, and how to read its metrics.
The design, the evaluation harness and every measured number live in
[`transcription_service/docs/speaker_diarization.md`](../transcription_service/docs/speaker_diarization.md);
this page is the operator's view.

## What it does

With diarization on, the `whisper-streaming` provider labels each caption
word with a stable per-session speaker (`spk_0`, `spk_1`, ...) and the
webapps render `Speaker 1:`, `Speaker 2:` in colours that stay readable on
the configured background. Diarization runs as a second worker-pool job on
its own worker process (`pyannote/speaker-diarization-community-1`, CPU by
default), so captions never wait for it: a label that is not ready when the
caption text appears arrives a moment later through a `speakers_update`
message and fills a placeholder slot in place. Speaker identity rests on
per-session voice embeddings kept in memory only (dropped 60 s after the
session ends; nothing is written to disk). Translated captions carry the same
labels, the transcript download writes one speaker turn per line, and the
screen-reader live region of the client and kiosk webapps announces each
speaker turn once.

Headline numbers (Linux CPU reference container, 4 CPUs, 8 GB, 10 s window,
5 s period; details and the per-meeting tables in the diarization doc):

| what | measured |
|---|---|
| diarization worker per session | 0.12 CPU cores, 0.7 GB RSS, real-time factor 0.135 (a pass costs 0.67 s of a 5 s period) |
| caption latency with diarization on against off | parity: on/off p50 ratio 0.86 over three alternating pairs; Whisper's own execution time is not higher beside the diarization worker |
| accuracy, 24-file set (16 AMI meetings + 8 VoxConverse files, 10 min each) | settled DER 0.266, speaker confusion 0.102, 1.06 labels on settled captions per real speaker, count within one on 19 of 24, 98 percent of final words labelled |
| label after the caption text appears | 0 s p50 and p95 (labels are usually ready before the text) |
| model load (warm, baked image) | 2.7 s in the container |
| sessions per 4-CPU container | one diarized session, limited by Whisper, not diarization |

## Images

Two production image families carry the feature, built from
[`transcription_service/Dockerfile_diarization`](../transcription_service/Dockerfile_diarization)
on top of the ordinary production image of the same device:

| `TRANSCRIPTION_DEVICE` in `.env` | image | base |
|---|---|---|
| `cpu-diarization` | `transcription-service-cpu-diarization` | `transcription-service-cpu` |
| `cuda-diarization`, `cuda128-diarization` | `transcription-service-cuda-diarization`, `transcription-service-cuda128-diarization` | the matching CUDA image |

The variant adds the `pyannote-diarization` dependency group (from the same
frozen `uv.lock`), `ffmpeg` (shared libraries pyannote's audio stack loads at
import), and the model itself: it is downloaded **once at build time** with a
HuggingFace token passed as a BuildKit secret, verified to load offline, and
stored at `/opt/scribear/models/pyannote/speaker-diarization-community-1`
(34 MB). The image names that directory in `SCRIBEAR_DIARIZATION_MODEL_DIR`,
and the pyannote context loads the default model from there. **A running
container needs no network access and no `HF_TOKEN`**; the token is never
written to a layer. The CPU variant is 2.2 GB against 1.3 GB for the base.

`compose.yml` needs no change: the image is selected by
`TRANSCRIPTION_DEVICE` exactly like the other variants, and the `/models`
bind mount (whisper and Silero weights) does not overlap the baked model.

### Building

The token's account must have accepted the model terms at
<https://huggingface.co/pyannote/speaker-diarization-community-1> (a gated
form; see "Licensing"). From the repository root:

```bash
export HUGGINGFACE_ACCESS_TOKEN=hf_...
./build-containers.sh v0.3.0        # builds every image; adds the diarization
                                    # variants when the token is set, skips
                                    # them with a notice otherwise
```

or by hand, from `transcription_service/`:

```bash
docker build -f Dockerfile_CPU -t scribear/transcription-service-cpu:dev .
export SCRIBEAR_BUILD_HF_TOKEN="$HUGGINGFACE_ACCESS_TOKEN"
DOCKER_BUILDKIT=1 docker build -f Dockerfile_diarization \
    --build-arg BASE_IMAGE=scribear/transcription-service-cpu:dev \
    --secret id=hf_token,env=SCRIBEAR_BUILD_HF_TOKEN \
    -t scribear/transcription-service-cpu-diarization:dev .
```

For a CUDA base add `--build-arg TORCH_EXTRA=silero-vad` (the CUDA images
install the CUDA torch through that extra; the CPU image uses
`silero-vad-cpu`, the default). A build without the secret fails at the bake
step, on purpose, rather than producing an image that fails at start-up.

### Verifying an image

```bash
cd transcription_service
scripts/check_diarization_image.sh scribear/transcription-service-cpu-diarization:dev
```

runs the image three times with `--network none` and no token: once with the
diarization template (must become ready, report the provider `OK`, log the
model loaded from the baked directory within the 10 s budget, and expose
`deviceFallbacks` on `/metrics/status`), once with a wrong
`diarization_context_tag` (must exit non-zero with the message naming the
tag), and once with the model directory removed (must exit non-zero naming
`SCRIBEAR_DIARIZATION_MODEL_DIR`). It mounts your host's whisper and Silero
caches, because with the network off the service cannot download them: the
same state a deployment is in after its first start.

## Configuration

Start from
[`provider_config.diarization.template.json`](provider_config.diarization.template.json):
two workers, whisper and Silero on worker 0, the pyannote context **on a
worker of its own** (worker 1), and `diarization_detector: true` on the
whisper provider. Point `PROVIDER_CONFIG_PATH` at your copy and set
`TRANSCRIPTION_PROVIDER_IDS` to the provider keys it defines.

A context on a shared worker is legal but warned about at start-up: a
worker runs one job at a time, so every diarization pass would hold the
caption job up for its duration.

### Provider keys (`providers.<key>.provider_config`)

Only the first two are needed; everything else shows its default. The
speaker-identity defaults were tuned on the AMI benchmark set and rarely
need changing.

| key | default | meaning |
|---|---|---|
| `diarization_detector` | `false` | Turns diarization on for this provider. Off, the provider registers exactly the upstream caption job and nothing else here runs. |
| `diarization_context_tag` | `"pyannote_diarization"` | Tag of the diarization context (`tags` of the pyannote context). Checked at start-up: a tag no live worker owns, or one that points at a non-diarization context, fails the service with a message. |
| `diarization_period_ms` | the caption `job_period_ms` (5000) | How often a diarization pass runs. 6000 lowers the worker's CPU by a sixth and raises label latency by about a second. |
| `diarization_window_sec` | `10` | Audio each pass looks at (the newest N seconds). 10 s is one segmentation chunk and the cheapest setting; longer windows cost superlinearly (see the window sweep in the diarization doc). Must exceed the period. |
| `diarization_edge_margin_sec` | `0.5` | The last half second of a window is left to the next pass, which sees it with context. |
| `diarization_label_timeout_sec` | `15` | A finalized caption that has waited this long for labels is settled with the labels it has (shown unattributed where none came). |
| `diarization_min_speakers`, `diarization_max_speakers` | unset | Optional bounds passed to the model. A lecture with occasional questions usually works well with `diarization_max_speakers: 4`. |
| `diarization_match_threshold` | `0.4` | Cosine similarity (plus overlap bonus) at or above which a voice attaches to the best-matching known speaker. |
| `diarization_new_speaker_threshold` | `0.3` | A voice mints a new speaker only when its best score against every known speaker is below this. |
| `diarization_attach_threshold` | `0.2` | A voice too short to mint attaches to the best speaker at or above this; below it the words stay unlabelled. |
| `diarization_min_mint_sec` | `2.5` | Speech a voice needs (in one pass or accumulated) before it can mint a label. |
| `diarization_max_session_speakers` | `32` | Bound on labels per session. |
| `diarization_merge_threshold` | `1.0` (off) | Two speakers whose centroids reach this similarity are merged. Measured: never removes a split label, merges real people below 0.5; leave off. |
| `diarization_merge_max_age_sec` | `20` | With merging on, only a speaker minted within this many seconds can be merged away (0 = any age). |
| `diarization_fragment_fold_sec` | `7.5` | A label minted within this many seconds whose audio the next pass re-labels as another speaker is folded into that speaker before its captions settle (a split of the instructor in a classroom). 7.5 s is one and a half periods: exactly the next pass. Scale with the period; `0` disables. Trade-off: in a many-speaker debate it can fold a real speaker's first turn (see Known limitations). |
| `diarization_fragment_fold_fraction` | `0.8` | Fraction of the fragment's audio in the next window that other speakers must cover for the fold. |
| `diarization_overlap_bonus` | `0.15` | Score added in proportion to the time overlap with the previous pass's labels. |
| `diarization_sustained_split_sec` | `0` (off) | A voice that keeps scoring between the new-speaker and the match threshold mints its own label after this many seconds of such evidence. Measured to add labels without finding a second person; leave off. |
| `diarization_recluster_period_sec` | `0` (off) | Re-cluster the session's track history with the model's PLDA/VBx clustering every this many seconds. Helps long sessions of a few long-speaking people, hurts the classroom; leave off unless measured. |
| `diarization_attach_gap_sec` | `1.0` | A word with no speech segment under it takes the nearest speaker within this gap; `0` disables. |
| `diarization_revision_margin` | `0.1` | A later pass replaces an earlier label on shared audio only when it is at least this much more confident (only words not yet finalized can change). |
| `diarization_reconnect_grace_sec` | `60` | How long a closed session's speaker memory is kept in memory for a reconnect with the same `session_uid`; `0` disables. |

### Context keys (`contexts[].context_config` of `pyannote-diarization`)

| key | default | meaning |
|---|---|---|
| `model` | `"pyannote/speaker-diarization-community-1"` | A HuggingFace id, or a local directory holding a pipeline `config.yaml`. In the diarization images the default id loads from the baked directory. |
| `device` | `"cpu"` | `cpu`, `cuda`, or `auto` (CUDA when torch sees a working device, else CPU). A CUDA request that is unavailable or fails its probe **falls back to the CPU with a warning** in the log and on `/metrics/status` (`deviceFallbacks`); it never fails the worker or a session. See "GPU deployments". |
| `token_env_var` | `"HUGGINGFACE_ACCESS_TOKEN"` | Where the token is read from when the model must come from HuggingFace (not needed with the baked model or `HF_HUB_OFFLINE=1` with a warm cache). |
| `num_threads` | `1` | Torch intra-op threads per pass. The embedding stage barely speeds up with more while the caption worker slows down; raise only with cores to spare. |
| `nice` | `10` | `os.nice` increment for the diarization worker, so the caption worker wins a contended core. `0` disables. |
| `segmentation_step` | `null` (model default, 1 s) | pyannote's segmentation step as a ratio of its 10 s window; irrelevant at the default 10 s window. |
| `local_speakers` | `true` | On a one-chunk window report the segmentation model's own speaker tracks (the pipeline's clustering collapses such a window to one speaker). |
| `overlap_aware` | `false` | Report overlapping turns instead of exclusive ones. Slightly better DER, more labels on screen; not the default. |
| `clustering_threshold` | `null` | Override of the pipeline's clustering threshold; only matters for windows longer than one chunk. |
| `shared_embeddings` | `true` | Run the embedding network once per window instead of once per speaker slot (same embeddings, a third of the cost). |

### Environment

| variable | where | meaning |
|---|---|---|
| `SCRIBEAR_DIARIZATION_MODEL_DIR` | set by the diarization images | Directory the default model is loaded from. Unset it to load from HuggingFace instead (token and network needed). A set variable pointing at an incomplete directory fails start-up with a message. |
| `HUGGINGFACE_ACCESS_TOKEN` / `HF_TOKEN` | only without a baked model | Token for the gated download; **not needed by the diarization images at runtime**. |
| `HF_HUB_OFFLINE=1` | optional | With a warm HuggingFace cache (`/models/hf`), load from it with no network and no token. |
| `MONITORING_DIARIZATION_UNCOVERED_RATIO`, `MONITORING_DIARIZATION_MIN_AUDIO_SECONDS`, `MONITORING_DIARIZATION_LAG_P95_MS` | `.env`, monitoring sidecar | Thresholds of the `diarizationBehindRule` alert (defaults 0.1, 30 s, 10 000 ms). |

No new **required** key is introduced: a stock deployment that does not opt
in changes nothing.

## Capacity: CPU and memory per session

Measured in the Linux CPU reference container (upstream's CPU image, 4 CPUs,
8 GB; Docker Desktop on an Apple M4, so a VM, not a server) with the
reference config; the diarization doc's "Phase 2 wrap-up results" and
"Phase 2a results" hold the full tables.

| process | CPU | peak RSS | notes |
|---|---|---|---|
| caption worker (whisper `base` + Silero), diarization off | about 1.0 core | 0.9 GB | upstream as shipped |
| caption worker, diarization on | about 1.0 core | 0.9 to 1.0 GB | unchanged job; parity measured over three alternating pairs |
| diarization worker, one session | **0.12 cores** | **0.7 GB** | real-time factor 0.135; pass 0.67 s mean, 0.82 s p95 |
| service tree, two workers | | about 2.2 GB | against 1.3 GB with one worker and no pyannote |

Per additional diarized session the diarization worker adds about 0.12 cores
and no further model memory (one model process serves every session on that
worker). The 4-CPU container sustains **one** diarized session, and the limit
is Whisper: at two sessions the caption worker drops a period per session
per period and caption latency triples, as it does without diarization. The
diarization worker only starts skipping audio at four sessions, and skips
show as `Speaker ?` captions and on `diarizationUncoveredSecondsTotal`,
never as caption latency. On hardware where Whisper serves N sessions the
diarization worker serves the same N up to about four before a second
diarization worker is worth adding (`worker_ids: [1, 2]`).

Memory over time: the two-hour soak in the same container
(`transcription_service/benchmarks/diarization/soak_service.py`) is the
leak check; its result is recorded in the diarization doc under "Phase 2c
production readiness".

## What happens when things fail

- **Start-up**: a missing `pyannote.audio` install, a diarization tag no
  worker owns or that names a non-diarization context, a token missing when
  the model must be downloaded, a baked model directory that is incomplete,
  or a pipeline that does not load each **fail the service at start-up**
  with a message naming the fix. None of them produces the old per-session
  `1011` reconnect loop. Readiness stays 503 and `/providers/health` reports
  the provider `DOWN` with the same reason when a worker is missing at
  runtime.
- **CUDA**: a missing or broken GPU falls back to the CPU with a warning;
  see "GPU deployments".
- **The diarization worker dies** (crash, OOM kill): the pool notices
  within a second, replaces the worker (same contexts; warm model load
  about 3 to 5 s), and every session registers a new diarization job seeded
  with its speaker memory and audio clock, so the same people keep the same
  labels. Captions continue uninterrupted meanwhile; words whose audio no
  pass covered settle as unattributed. `/metrics/status` counts the
  replacement in `workerRestarts`. The caption worker is not re-registered
  by sessions today (upstream's path); its replacement serves new sessions.
- **A diarization pass raises**: the pass is counted in
  `diarizationFailedTotal`, that audio stays unlabelled, captions continue.
- **Diarization falls behind** (CPU starved): audio is skipped, never
  queued; `diarizationUncoveredSecondsTotal` and
  `diarizationDroppedPeriodsTotal` rise and the sidecar's
  `diarizationBehindRule` fires. Caption latency is unaffected by design.

## GPU deployments

Set `"device": "auto"` (or `"cuda"`) in the pyannote context of a
`cuda-diarization` image deployment under the GPU overlay
(`compose.gpu.yml`). On a working GPU the worker logs `resolved to CUDA`,
`/metrics/status` reports `providerDevice` for the diarization context's
provider as `cuda` and `deviceFallbacks` stays empty. On a host where the
driver is missing or the probe fails, the worker logs `falling back to CPU`
with the reason, `deviceFallbacks` names the context tag with
`configured_device`, `device` and `reason`, and labels keep coming from the
CPU; nothing restarts and no session is lost.

Unit tests cover the selection and both fallbacks
(`tests/unit/shared/utils/diarization_backend/select_device_test.py`,
`tests/unit/transcription_contexts/pyannote_diarization_context_test.py`).
The CUDA path itself was **not exercised in this release's development
environment** (an Apple laptop; no NVIDIA GPU). To verify it on NCSA Delta:

1. On a GPU node (`srun --partition=gpuA40x4 --gpus=1 ...` or an Open
   OnDemand Jupyter session with a GPU), clone the fork, `cd
   transcription_service`, `uv sync --extra faster-whisper --extra
   silero-vad --extra pyannote-diarization` (the CUDA torch), export
   `HUGGINGFACE_ACCESS_TOKEN`.
2. Start the service with a copy of `provider_config.diarization.template.json`
   whose whisper context has `"device": "cuda"` and whose pyannote context
   has `"device": "auto"`, and check the start-up log for
   `Diarization device 'auto' resolved to CUDA` and
   `loaded successfully in N s on cuda`.
3. Stream a two-speaker 16 kHz WAV with
   `tests/manual/transcription_stream_file_client.py` and confirm labels; on
   `/metrics/status` confirm `providerDevice` says `cuda` and
   `deviceFallbacks` is `{}`. `nvidia-smi` shows the worker process on the
   GPU.
4. Fallback: start again with `CUDA_VISIBLE_DEVICES=` (empty) in the
   environment; the log must show `falling back to CPU`, `deviceFallbacks`
   must name `pyannote_diarization`, and a streamed session must still get
   labels.
5. Optional: `make benchmark_diarization_caption` with
   `--context-set pyannote-diarization.device=cuda` records the pass cost on
   the GPU for the diarization doc.

The backend interface (`src/shared/utils/diarization_backend/`) is what a
GPU-native model such as NVIDIA's Streaming Sortformer would implement as a
second context; nothing in the provider or the job is pyannote-specific.

## Metrics, panels and alerts

`/metrics/status` (read with `TRANSCRIPTION_METRICS_KEY`) exports, keyed by
the caption provider's key and all empty when diarization is off:

| field | read it as |
|---|---|
| `diarizationRunsTotal`, `diarizationSecondsTotal` | passes and their wall time; seconds / runs is the pass cost (0.67 s on the reference container) |
| `diarizationAudioSecondsTotal` | audio received by the diarization job; seconds / audio is the real-time factor (0.135) |
| `diarizationFailedTotal` | passes that raised; captions unaffected, that audio unlabelled |
| `diarizationLabelsMintedTotal` | session labels minted; far above the people in the room means over-splitting |
| `diarizationUncoveredSecondsTotal` | audio no pass covered because the job fell behind: those words show `Speaker ?` |
| `diarizationDroppedPeriodsTotal` | periods skipped because the previous pass overran |
| `reconcilerSecondsTotal` | reconciler cost (microseconds per pass) |
| histograms `diarizationExecutionMs`, `diarizationLagMs`, `diarizationRtf` | pass cost, age of the newest labelled audio when its labels were ready (about 1 s), RTF per pass |
| `providerDevice` | the device each provider's context runs on; with `auto` the device actually chosen |
| `deviceFallbacks` | contexts running on the CPU although CUDA was configured, with the reason; empty when healthy |
| `workerRestarts` | worker processes the pool replaced after they died, by worker id; zero when healthy |

The caption series (`asr*`) describe the caption job alone: the metrics
registry folds the diarization job's executions apart by its observer
label, so a slow diarization pass can never appear as a dropped caption
period.

The monitoring sidecar polls all of these (optional fields, so an older
service still validates) and publishes them as `scribear_diarization_*`,
with `scribear_diarization_supported` as the "service reports it at all"
guard. The Grafana fleet dashboard (`deployment/monitoring`, see its README
for turning monitoring on) has two diarization panels:

- **Speaker diarization: pass cost and label lag (p95)**: pass cost against
  the job period (healthy: a fraction of the period; the reference
  container sits at 0.67 s of 5 s) and the p95 lag (healthy: about 1 s;
  rising lag means the worker is starved).
- **Speaker diarization: audio skipped, dropped periods, failed passes**:
  all three flat at zero in a healthy deployment. Audio skipped means
  `Speaker ?` captions; failed passes mean a model error worth the service
  log.

The alert `diarizationBehindRule` warns when more than
`MONITORING_DIARIZATION_UNCOVERED_RATIO` (0.1) of the last window's audio was
skipped (once at least `MONITORING_DIARIZATION_MIN_AUDIO_SECONDS` of audio
arrived) or the p95 lag exceeds `MONITORING_DIARIZATION_LAG_P95_MS`
(10 000 ms), and is critical when audio keeps arriving but no pass
completes at all (the hung-worker case). `workerRestarts` and
`deviceFallbacks` are on `/metrics/status` for a dashboard query; no alert
rule reads them yet.

## Licensing and privacy

| component | license | obligations |
|---|---|---|
| `pyannote/speaker-diarization-community-1` (segmentation and embedding models, PLDA) | **CC BY 4.0**, gated | The downloading account accepts the terms (contact details, consent to occasional email from pyannote). **Attribution is required**: name the model and pyannoteAI in the product's documentation or about screen. This repository does so in `README.md`, `transcription_service/docs/speaker_diarization.md`, this page and the image label `org.scribear.diarization.model-license`. Commercial use is allowed. Citation: Bredin, "pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe", Interspeech 2023; Plaquet and Bredin, "Powerset multi-class cross entropy loss for neural speaker diarization", Interspeech 2023. |
| `pyannote.audio` 4.x (library) | MIT | notice in the dependency tree |
| WeSpeaker ResNet34 embedding weights (inside community-1) | distributed under the model's CC BY 4.0 | covered by the attribution above |
| faster-whisper, CTranslate2, `Systran/faster-whisper-*` weights | MIT | unchanged from upstream |
| Silero VAD | MIT | unchanged from upstream |
| torch, torchaudio, torchcodec | BSD-3 | unchanged from upstream |
| ffmpeg (shared libraries in the diarization image, Debian package) | LGPL 2.1+ / GPL 2+ depending on the Debian build | dynamically linked, unmodified; the Debian package's copyright file travels in the image |

Privacy: no audio and no embeddings are written to disk. Audio lives in the
workers' in-memory buffers; the session's speaker centroids are voiceprints
(biometric data under GDPR and Illinois BIPA) and live only in the service
process, for the session plus the 60 s reconnect grace, then are dropped.
Speaker labels travel to every viewer in the room and into node-server's
transcript bus; the session manager persists nothing about them. The
transcript download (client webapp) includes the labels the viewer saw.

## Troubleshooting

| start-up message | fix |
|---|---|
| `pyannote.audio is not installed but a pyannote-diarization context is configured` | use a `*-diarization` image, or remove the context and turn `diarization_detector` off |
| `no live worker owns a context tagged '...'` | the provider's `diarization_context_tag` must equal one of the pyannote context's `tags`, on a worker in `worker_ids` |
| `diarization_context_tag '...' resolves to ['FasterWhisperContext'], which cannot diarize` | the tag names the wrong context |
| `Environment variable 'HUGGINGFACE_ACCESS_TOKEN' must be set to load ...` | the image has no baked model: use a diarization image, or set the token and allow network access |
| `SCRIBEAR_DIARIZATION_MODEL_DIR='...' is set but holds no config.yaml` | the bake failed or the directory was replaced: rebuild the image (the bake step fails loudly without a token) |
| `Could not load ... from HuggingFace: accept the model terms ...` | the token's account has not accepted the gated terms, or the host cannot reach huggingface.co |
| `Every worker owning the diarization context also runs captions` (warning) | give the pyannote context `worker_ids` of its own and `num_workers: 2` |
| `Diarization device 'cuda' requested but CUDA is not available; falling back to CPU` (warning) | labels work from the CPU; fix the driver or the GPU overlay when you want the GPU |
