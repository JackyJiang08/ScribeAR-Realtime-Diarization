# Speaker diarization

The whisper-streaming provider can optionally label each transcribed word
with a speaker (`spk_0`, `spk_1`, ...), so clients can render who said what.
Diarization runs on CPU by default and is fully optional: deployments that do
not configure it behave exactly as before.

## How it works

Each job tick the provider already re-transcribes the rolling audio buffer.
With diarization enabled it additionally:

1. Runs a pyannote speaker diarization pipeline over the same buffer,
   producing raw per-run speaker ranges (`PyannoteDiarizationContext`).
2. Maps the raw labels to stable session-wide labels. Diarization models
   assign arbitrary labels per run, so the same voice could flip between
   `SPEAKER_00` and `SPEAKER_01` on consecutive ticks; `SpeakerReconciler`
   fixes this by matching each run's labels to the previous run's labels
   through overlap voting on the shared window.
3. Assigns each transcribed word the speaker range it overlaps the most.
4. Emits labels through the existing `TranscriptionSequence` as a `speakers`
   array aligned with `text`. Words with no attributable speaker carry `null`.

Speaker labels do not participate in Local Agreement finalization: a label
flip never delays caption finalization.

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

3. Add the context and enable the provider flag in your provider config
   (see `provider_config.template.json`):

   ```json
   {
       "context_uid": "pyannote-diarization",
       "worker_ids": [0],
       "tags": ["pyannote_diarization"],
       "context_config": {
           "model": "pyannote/speaker-diarization-community-1",
           "device": "cpu",
           "token_env_var": "HUGGINGFACE_ACCESS_TOKEN"
       }
   }
   ```

   and in the whisper-streaming `provider_config`:

   ```json
   "diarization_detector": true,
   "diarization_context_tag": "pyannote_diarization"
   ```

   Optional bounds: `diarization_min_speakers`, `diarization_max_speakers`
   (a lecture with occasional questions usually works well with
   `diarization_max_speakers: 4`).

## Testing

1. Unit tests (no models needed):

   ```bash
   make install_dev_cpu
   make test_unit
   ```

2. End-to-end against a locally running service. Create a `.env`
   (see repository docs) pointing `PROVIDER_CONFIG_PATH` at a config with
   `diarization_detector: true`, export `HUGGINGFACE_ACCESS_TOKEN`, start
   the service with `make dev`, then stream a recording into it:

   ```bash
   uv run python tests/manual/transcription_stream_file_client.py \
       --audio sample.wav --api-key <API_KEY from .env>
   ```

   Finalized lines print with inline `[spk_N]` markers on speaker changes.
   Use a 16 kHz mono WAV with at least two speakers; the first diarization
   run downloads the pyannote model, so expect a slow first tick.

3. Performance measurement on target hardware: `benchmarks/diarization/`.

## Performance notes

Diarization shares the worker's CPU budget with Whisper. If ticks start
exceeding `job_period_ms`, pin the diarization context to its own worker
process: raise `num_workers` and give the `pyannote-diarization` context a
dedicated entry in `worker_ids`. Measure real hardware with
`benchmarks/diarization/` before enabling in production.

## Known limitations

- Labels can still drift after long silences or when the rolling buffer is
  force-purged past the last diarized window; the reconciler then mints a
  fresh label for what may be the same voice. Fixing this properly requires
  speaker-embedding memory, which is planned as a follow-up.
- Words already finalized keep the label they were finalized with; later,
  better speaker evidence does not retroactively update them.
- Accuracy degrades with heavily overlapping speech and far-field audio.
- GPU deployments can set `"device": "cuda"` in the context config; a
  streaming-native alternative (NVIDIA Streaming Sortformer) can be added as
  another context implementation without touching the provider wiring.
