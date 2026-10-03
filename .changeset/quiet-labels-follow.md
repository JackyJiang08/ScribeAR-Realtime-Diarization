---
'@scribear/transcription-service-schema': minor
'@scribear/node-server-schema': minor
'@scribear/node-server': minor
'@scribear/transcription-content-store': minor
'@scribear/transcription-display-ui': minor
'@scribear/client-webapp': minor
'@scribear/kiosk-webapp': minor
'@scribear/monitoring-sidecar': minor
---

Speaker labels no longer hold captions back (diarization Phase 2a).

The transcription service runs diarization as its own worker-pool job on its
own worker; captions are emitted the moment Whisper produces them and speaker
labels follow through a new, backward-compatible `speakers_update` server
message (`speakersUpdate` on node-server) that names the finalized sequence by
the `sequence_id` / `sequenceId` it was sent with. Both fields are optional in
the shared schemas; a client that ignores the message still shows every
caption, and nothing changes when diarization is off.

The content store applies late labels by filling unlabelled words only (a label
already shown never changes), and the caption display reserves a fixed-width
`Speaker ?` slot at the start of each labelled line that fills in place, so no
text moves and no live region re-announces when a label arrives.

The monitoring sidecar polls the service's diarization counters and histograms
into `scribear_diarization_*` series (with a `scribear_diarization_supported`
guard), adds `diarizationBehindRule` (skipped audio, high label lag, stalled
job) with three new `ALERT_DIARIZATION_*` thresholds, and the Grafana fleet
dashboard gains two diarization panels.
