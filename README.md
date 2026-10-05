> **Fork notes — JackyJiang08/ScribeAR-Realtime-Diarization**
>
> This fork of [scribear/scribear](https://github.com/scribear/scribear) adds
> **real-time speaker diarization** as a deployable, optional feature of the
> transcription service: the whisper-streaming provider labels each caption
> word with a stable per-session speaker (`spk_0`, `spk_1`, ...), and the
> client, kiosk and translated-caption views render `Speaker 1:`,
> `Speaker 2:` in colours that stay readable on every theme. Everything else
> tracks upstream `staging`. Off by default; a deployment that does not opt
> in is unchanged.
>
> **What it does.** Diarization runs as its own worker-pool job on its own
> worker process, so captions never wait for it: labels ride on the caption
> when they are ready (almost always) and otherwise follow through a
> `speakers_update` message that fills a label slot in place. Speaker
> identity rests on per-session voice embeddings kept in memory only (dropped
> 60 s after the session ends; nothing touches disk). Translated captions
> carry the same labels, the transcript download writes one speaker turn per
> line, and the client and kiosk webapps' screen-reader live region now
> announces caption paragraphs with each speaker named once per turn (a gap
> inherited from upstream, where only the standalone app ever fed that
> region). A dead diarization worker is replaced by the pool within seconds
> and the session's labels resume with the same identities; CUDA is probed
> and falls back to the CPU with a warning rather than failing a session; a
> bad diarization config fails at start-up with the fix in the message.
>
> **Headline numbers** (Linux CPU reference container, 4 CPUs, 8 GB; 10 s
> window, 5 s period; every table in the diarization doc): the diarization
> worker costs 0.12 cores and 0.7 GB per container at a real-time factor of
> 0.135; caption latency with diarization on is at parity with off (on/off
> p50 ratio 0.86 over three alternating pairs); on the 24-file evaluation set
> (16 AMI meetings and 8 VoxConverse files, 10 min each) settled DER 0.266,
> speaker confusion 0.102, 1.06 labels on settled captions per real speaker,
> speaker count exact on 11 and within one on 19 of 24, 98 percent of final
> words labelled, labels 0 s after the caption text; warm model load 2.7 s
> from the baked image. A two-hour soak in the same container on a quiet host, with the
> diarization worker killed at 60 min, grew the service's memory 1.5
> percent, kept the same four people under the same four labels in every
> 15-minute bin, kept every diarization counter clean (no audio skipped,
> no failed pass, no label changed after sending), replaced the worker in
> 4.2 s with the session's job back 5 s after the kill (the first label on
> new audio reached the screen 20 s after it, one caption latency plus one
> diarization period later), and dropped 83 s of caption audio in its
> first hour against 75 s for the same hour with diarization off; the final gate run passed
> every gated metric against the previous baseline with a clean caption
> stream (on 4.57 s against off 4.74 s p50) and the whole-file offline
> pass included (model ceiling DER 0.190, confusion 0.025 on the set), and
> is now the gate baseline. Still open and documented: confusion 0.10
> against the 0.08 target (the clustering context, not the model), the
> set's speaker count within one on 19 of 24 (the fragment fold costs a
> five-person debate two labels), far-field under-counting and clean
> many-speaker panels, a single-run dropped-period count of 10 against 5
> with diarization on (inside the off configuration's own spread), the
> soak's 8 s of extra dropped audio in an hour, and the 20 s a viewer waits
> for the first label after a diarization worker dies.
>
> **Enable.** Deploy the `transcription-service-<device>-diarization` image
> (`TRANSCRIPTION_DEVICE=cpu-diarization` in `deployment/.env`; the model is
> baked in at build time, the container needs no network and no HuggingFace
> token), point `PROVIDER_CONFIG_PATH` at a copy of
> [`deployment/provider_config.diarization.template.json`](deployment/provider_config.diarization.template.json)
> and set `TRANSCRIPTION_PROVIDER_IDS` to its keys. For development:
> `uv sync --extra pyannote-diarization`, accept the gated model terms,
> export `HUGGINGFACE_ACCESS_TOKEN`, add the `pyannote-diarization` context
> on a worker of its own (`num_workers: 2`) and set
> `"diarization_detector": true` on the whisper provider. Judge changes with
> `make benchmark_diarization_gate_docker` and
> `make benchmark_diarization_acceptance`.
>
> **Model credit and license.** Speaker labels come from
> [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
> by pyannoteAI (Hervé Bredin and contributors), licensed **CC BY 4.0** and
> distributed behind a gated access form, through the MIT-licensed
> `pyannote.audio` library. Every model the service can load, with its
> license and terms, is listed in `deployment/DIARIZATION.md` ("Licensing
> and privacy").
>
> **Read next.** Deployment guide (images, every configuration key, cost per
> session, verification, metrics and Grafana panels, licensing):
> [`deployment/DIARIZATION.md`](deployment/DIARIZATION.md). Design,
> measurements and **Known limitations** (far-field under-counting, clean
> many-speaker panels, the fragment fold's trade-off, confusion against the
> offline pipeline, what a dead worker costs):
> [`transcription_service/docs/speaker_diarization.md`](transcription_service/docs/speaker_diarization.md#known-limitations).
> Issues worth raising upstream: [`docs/upstream_issue_drafts.md`](docs/upstream_issue_drafts.md).
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
