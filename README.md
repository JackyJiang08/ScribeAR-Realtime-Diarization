> **Fork notes — JackyJiang08/ScribeAR-Realtime-Diarization**
>
> This fork of [scribear/scribear](https://github.com/scribear/scribear) adds
> **real-time speaker diarization**: the whisper-streaming provider can label
> each word with a stable speaker (`spk_0`, `spk_1`, ...) and the viewer renders
> colored `Speaker N:` labels. Everything else tracks upstream `staging`.
>
> **Status (2026-10-04, Phase 2c):** working end to end, optional, off by
> default. Captions never wait for diarization: it runs as its own worker-pool
> job on its own worker and labels reach already-shown captions through a
> `speakers_update` message that fills a label slot in place. Speaker
> identity rests on per-session embedding memory (in memory only, kept 60 s
> for a reconnect). Phase 2c re-examined speaker counting on a broader set:
> all 16 AMI test meetings plus an 8-file VoxConverse subset (1 to 8
> speakers), full 10 minutes each, and a synthetic classroom case (one
> instructor 83 percent of the time, nine 5 to 15 s questions, one voice
> under 5 dB pink noise). Measured on upstream's CPU image with 4 CPUs:
> 1.17 labels per real speaker (band 0.8 to 1.2), speaker count within one
> on 20 of 24 meetings, 8 of 9 classroom questions labelled as someone
> other than the instructor (all three noisy ones), settled DER 0.27,
> 99.6 percent of finalized words labelled, captions with diarization on
> at 5.7 s p50 against 5.8 s off, diarization worker 0.27 cores, 0.87 GB,
> real-time factor 0.30. The typical error is one far-field speaker split
> into fragments, not two people merged; the two merge-guards that were
> built (sustained-voice split, PLDA/VBx re-clustering of the session
> memory) only trade confusion for more labels and ship disabled. Missed:
> confusion 0.10 over the whole set against the 0.08 target, RTF 0.30
> against the 0.25 margin target (a 6 s period reaches 0.25 but puts label
> latency p95 at 12.7 s, so 5 s stays), and two clean VoxConverse panels
> (six and three speakers) that end with two labels although offline
> pyannote separates every voice: the one real pipeline under-count left.
> Full tables, per-meeting counts and caveats are in the diarization doc
> linked below.
>
> **Enable:** `uv sync --extra pyannote-diarization`, accept the gated
> `pyannote/speaker-diarization-community-1` terms, export
> `HUGGINGFACE_ACCESS_TOKEN`, add the `pyannote-diarization` context to
> `provider_config.json` on a worker of its own (`num_workers: 2`) and set
> `"diarization_detector": true` on the whisper provider. Judge changes with
> `make benchmark_diarization_gate_docker` and
> `make benchmark_diarization_acceptance`.
>
> Design, configuration, testing and benchmark details:
> [`transcription_service/docs/speaker_diarization.md`](transcription_service/docs/speaker_diarization.md).
> Upstream sync log: [`docs/upstream_sync_2026-10.md`](docs/upstream_sync_2026-10.md).

# ScribeAR

Self-hosted, real-time transcription. This monorepo (`scribear/scribear`) contains everything behind [ScribeAR](https://scribear.illinois.edu/v/index.html): the speech-to-text service, the proxy/session backend, the Postgres schema, and the client/kiosk/standalone webapps.

Full architecture, protocols, and API reference live in the **[wiki](https://github.com/scribear/scribear/wiki)** — this README is just a map to get you to the right page.

## Repo layout

```
apps/
  client-webapp/       # viewer — joins a session via a join code, receives transcripts
  kiosk-webapp/         # source — the device sending audio for a room, shows a join QR code
  standalone-webapp/    # all-in-one viewer+source app, no kiosk/client split
  node-server/           # proxies kiosk/client websockets to transcription-service
  session-manager/       # devices, rooms, sessions, auth — issues session tokens
  admin-webapp/         # IT admin console SPA (rooms, devices, kiosks) — talks only to admin-server
  admin-server/          # admin BFF — holds the Session Manager admin key, authenticates staff, proxies + audits
infra/
  scribear-db/           # Postgres schema + migrations
  scribear-nginx/        # reverse proxy used in the deployment stack
libs/
  clients/               # typed clients (session-manager, node-server, transcription-service, ...)
  schemas/                # shared request/response schemas
  store/                  # shared Redux slices used by the webapps
  ui/                     # shared React components used by the webapps
transcription_service/   # Python: the actual speech-to-text models (faster-whisper, CPU/CUDA)
deployment/               # Docker Compose stack for running the full system
```

## Start here, by audience

**New here / just curious / thinking about joining** — start at the wiki [Home](https://github.com/scribear/scribear/wiki/Home) page for the full architecture picture, [`RELEASING.md`](RELEASING.md) for how branches (`staging`/`main`) and releases work, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the repo-specific rules this codebase has learned the hard way (monitoring guards, long-lived-branch merges, protocol and schema compatibility, accessibility traps).

**Frontend developers** (client-webapp, kiosk-webapp, standalone-webapp, `libs/ui`, `libs/store`) — see the wiki [Developing Frontend](https://github.com/scribear/scribear/wiki/Developing-Frontend) page to get an app running locally, and [Connecting From Frontend](https://github.com/scribear/scribear/wiki/Connecting-From-Frontend) for the session-token/websocket protocol these apps speak.

**Backend developers** (node-server, session-manager, transcription-service, scribear-db) — see [Developing Node Server](https://github.com/scribear/scribear/wiki/Developing-Node-Server), [Developing Session Manager](https://github.com/scribear/scribear/wiki/Developing-Session-Manager), and [Developing Transcription Service](https://github.com/scribear/scribear/wiki/Developing-Transcription-Service), plus the full [Documentation](https://github.com/scribear/scribear/wiki/Documentation) page for the API/protocol/config reference.

**Deployment & production** — see the wiki [Deployment](https://github.com/scribear/scribear/wiki/Deployment) page for the Docker Compose stack, and [`RELEASING.md`](RELEASING.md#container-tags) for how image tags map to branches (`staging` → `staging`/`staging-<sha>`, `main` → `latest`/`v<version>`). **Upgrading an existing deployment — read [`deployment/UPGRADING.md`](deployment/UPGRADING.md) first**: `deployment/.env` is untracked and does not update when you pull, so releases that add a required key will refuse to start until you add it. Evaluating the stack on a staging box — see [`deployment/monitoring/README.md`](deployment/monitoring/README.md) for the opt-in, zero-click Prometheus + Grafana fleet dashboard.

**AI coding agents / LLMs** — read this file, [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`RELEASING.md`](RELEASING.md) first, then the wiki [Documentation](https://github.com/scribear/scribear/wiki/Documentation) page before making assumptions about API shapes, message protocols, or config — it's the authoritative machine-actionable reference. This is an npm workspace monorepo: `npm install` at the root installs everything, and `npm run build|lint|format|test:unit|test:integration` at the root run across all workspaces (`--workspace <path>` to scope to one). CI/CD is in `.github/workflows/` (`node-ci`/`node-cd`, `python-ci`/`python-cd`) and the composite actions it uses are in `.github/actions/`.

## Full wiki index

* [Home](https://github.com/scribear/scribear/wiki/Home) — architecture overview
* [Deployment](https://github.com/scribear/scribear/wiki/Deployment) — run the full stack with Docker
* [Connecting From Frontend](https://github.com/scribear/scribear/wiki/Connecting-From-Frontend) — session tokens and the node-server websocket protocol
* [Developing Frontend](https://github.com/scribear/scribear/wiki/Developing-Frontend) — client/kiosk/standalone webapps
* [Developing Node Server](https://github.com/scribear/scribear/wiki/Developing-Node-Server)
* [Developing Session Manager](https://github.com/scribear/scribear/wiki/Developing-Session-Manager)
* [Developing Transcription Service](https://github.com/scribear/scribear/wiki/Developing-Transcription-Service)
* [Admin Website](https://github.com/scribear/scribear/wiki/Admin-Website) — operator guide for the IT admin console
* [Developing Admin](https://github.com/scribear/scribear/wiki/Developing-Admin)
* [Documentation](https://github.com/scribear/scribear/wiki/Documentation) — full API/protocol/config reference
* [ScribeAR Multi Tenancy HLD](https://github.com/scribear/scribear/wiki/ScribeAR-Multi-Tenency-HLD) — historical design notes, background only
