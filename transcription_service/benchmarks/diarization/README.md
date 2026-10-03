# Diarization evaluation harness

Everything that judges a Phase 2 diarization change lives here. The full
guide, including what every metric means and how to read a report, is in
[`../../docs/speaker_diarization.md`](../../docs/speaker_diarization.md)
(section "Evaluation harness"). This file is the map.

| file | purpose |
|---|---|
| `configs/reference_provider_config.json` | **Reference config** for every run: upstream's shipped deployment defaults (whisper `base`, no VAD, 5 s period, 30 s buffer) plus the pyannote context and `diarization_detector: true`. The harness derives the diarization-off variant (context removed) so "off" is exactly upstream as shipped. |
| `configs/dev_vad_provider_config.json` | Secondary, clearly labelled: the fork's old dev config with `vad_detector: true`. Reproduces the audit's 27 to 30 s latency; never gated. |
| `run_suite.py` | The standard suite: hygiene, warm-up, replay benchmark, end-to-end caption latency on/off, one JSON report with a flat `key_metrics` block. |
| `benchmark_baseline.py` | Replay benchmark: offline + streaming DER/JER, speaker-count error, label latency, flips, labels minted, per-stage timing, modelled lag, memory. |
| `caption_latency.py` | Runs the real service and streams audio at it: chunk-id latency (primary, node-server's method) and word latency (secondary), service counters, RSS. |
| `compare_baseline.py` | Regression gate against `baselines/<environment>.json` with `baselines/gate_rules.json`. |
| `baselines/` | Committed reference reports (the Phase 2 starting point) and the gate rules. |
| `docker/` | Linux CPU reference environment: upstream's `Dockerfile_CPU` image plus the pyannote extra, run with `--cpus 4 --memory 8g` by default. |
| `warmup.py` | One model load of whisper, Silero and pyannote before timed runs. |
| `hard_cases.json`, `select_hard_cases.py`, `prepare_hard_cases.py` | Hard-case set (overlap, short turns, return after a long gap, four speakers, background noise), its selection method and its source documentation. |
| `prepare_soak.py`, `soak.py` | 60 min / 2 h soak stream from consecutive AMI meetings and the harness that tracks drift, label swaps and memory growth. |
| `prepare_ami_baseline.sh`, `ami_download.py` | Download and crop the AMI meetings and references into `data/` (gitignored). |
| `bench_common.py` | Shared helpers: hygiene, resource limits, RSS, percentiles, loaders. |
| `benchmark_diarization.py` | Older speed-only comparison of pyannote vs NVIDIA Sortformer. Kept for GPU experiments. |
| `results/` | Per-run reports (gitignored except the two historical reports). |

Quick start (from `transcription_service/`, token exported, ffmpeg installed):

```bash
make benchmark_diarization_suite            # native, quick dev loop
make benchmark_diarization_suite_docker     # Linux CPU reference, 4 CPUs / 8 GB
make benchmark_diarization_gate             # suite + fail on regression
make benchmark_diarization_hardcases
make benchmark_diarization_soak SOAK_MINUTES=60
```

Never commit audio. `data/` is gitignored; every source is documented in
`hard_cases.json`, `prepare_soak.py` and the docs.
