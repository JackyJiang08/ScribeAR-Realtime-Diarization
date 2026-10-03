.DEFAULT_GOAL := start

install_uv:
	@command -v uv >/dev/null 2>&1 || pip install uv

install_dev: install_uv
	uv sync --extra dev --extra faster-whisper --extra silero-vad --extra pyannote-diarization

install_dev_cpu: install_uv
	uv sync --extra dev --extra faster-whisper --extra silero-vad-cpu --extra pyannote-diarization

install: install_uv
	uv sync

# Run application in prod mode
start:
	uv run python src/index.py

# Run application in dev mode (hot reload + pretty print enabled)
dev:
	uv run watchfiles "python src/index.py --dev"

# Formatting without editing and run linter
format:
	uv run isort --check ./src ./tests
	uv run black --check ./src ./tests

# Format source code
format_fix:
	uv run isort ./src ./tests
	uv run black ./src ./tests

# Run linter
# Tests waive two rules that are pytest idiom rather than defects:
# redefined-outer-name (a test taking its own fixture as a parameter) and
# duplicate-code (independent suites each standing up the same fixtures and
# handshakes). Sharing that boilerplate across suites to satisfy R0801 would
# couple tests that are meant to be readable and fail in isolation.
lint:
	uv run pylint -j 0 --recursive=y --fail-under=10 ./src
	uv run pylint -j 0 --recursive=y --fail-under=10 --disable=redefined-outer-name,duplicate-code ./tests

# Run tests in dev mode (hot reload)
test_dev:
	uv run watchfiles "pytest tests/unit tests/integration"

# Run tests
test_unit:
	uv run coverage run -m pytest tests/unit && uv run coverage xml --include=src/* && uv run coverage report --include=src/*

test_integration:
	uv run coverage run -m pytest tests/integration && uv run coverage xml --include=src/* && uv run coverage report --include=src/*

# ---------------------------------------------------------------------------
# Speaker diarization evaluation (benchmarks/diarization/, documented in
# docs/speaker_diarization.md). Needs HUGGINGFACE_ACCESS_TOKEN or HF_TOKEN for
# the gated pyannote model and ffmpeg for data preparation. The first run
# downloads ~110 MB of AMI audio into benchmarks/diarization/data (gitignored).
#
#   RESULT=...      report path (default: results/<date>_<target>.json)
#   STREAM_SEC=...  streaming replay length per file (default 120, 0 = full)
#   CAPTION_SEC=... seconds streamed in the end-to-end caption probe (180)
#   LABEL=...       free-text label stored in the report
#   SUITE_ARGS=...  extra run_suite.py flags, e.g. --with-dev-vad --skip-offline
#   BENCH_CPUS / BENCH_MEMORY_GB  container limits for the *_docker targets (4 / 8)
BENCH_DIR := benchmarks/diarization
DATE := $(shell date +%Y%m%d)
RESULT ?= $(BENCH_DIR)/results/$(DATE)_suite.json
STREAM_SEC ?= 120
CAPTION_SEC ?= 180
LABEL ?=
SUITE_ARGS ?=
GATE_BASELINE ?= auto

# Download and crop the AMI meetings for the standard set
benchmark_diarization_data:
	$(BENCH_DIR)/prepare_ami_baseline.sh 10

# Replay benchmark only (offline + streaming DER/JER, label metrics, tick cost).
# Kept as the quick accuracy loop; the suite below includes it.
benchmark_diarization_baseline: benchmark_diarization_data
	uv run python $(BENCH_DIR)/benchmark_baseline.py \
		--data $(BENCH_DIR)/data/ami --suffix _10min \
		--stream-sec $(STREAM_SEC) --label "$(LABEL)" \
		--out $(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_baseline.json,$(RESULT))

# End-to-end caption latency only, diarization on, reference config
benchmark_diarization_caption: benchmark_diarization_data
	uv run python $(BENCH_DIR)/caption_latency.py \
		--provider-config $(BENCH_DIR)/configs/reference_provider_config.json \
		--audio $(BENCH_DIR)/data/ami/ES2004a_10min.wav \
		--diarization on --seconds $(CAPTION_SEC) --label "$(LABEL)" \
		--out $(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_caption_on.json,$(RESULT))

# The standard suite, natively (quick dev loop on this machine)
benchmark_diarization_suite: benchmark_diarization_data
	uv run python $(BENCH_DIR)/run_suite.py \
		--stream-sec $(STREAM_SEC) --caption-seconds $(CAPTION_SEC) \
		--label "$(LABEL)" --out $(RESULT) $(SUITE_ARGS)

# The standard suite inside upstream's Linux CPU image with fixed limits
# (the reference environment). Report lands in results/ like the native one.
benchmark_diarization_suite_docker: benchmark_diarization_data
	$(BENCH_DIR)/docker/run_in_docker.sh \
		--stream-sec $(STREAM_SEC) --caption-seconds $(CAPTION_SEC) \
		--label "$(LABEL)" --out $(subst $(BENCH_DIR)/,/app/$(BENCH_DIR)/,$(RESULT)) $(SUITE_ARGS)

# Regression gate: run the suite, then fail if any gated key metric is worse
# than the committed baseline for this environment beyond its tolerance
# (baselines/gate_rules.json). GATE_BASELINE=path overrides the auto pick.
benchmark_diarization_gate: benchmark_diarization_suite
	uv run python $(BENCH_DIR)/compare_baseline.py --report $(RESULT) --baseline $(GATE_BASELINE)

benchmark_diarization_gate_docker: benchmark_diarization_suite_docker
	uv run python $(BENCH_DIR)/compare_baseline.py --report $(RESULT) --baseline $(GATE_BASELINE)

# Hard cases: overlap, short turns, return after a long gap, 4 speakers,
# background noise (hard_cases.json). Streams every case in full.
benchmark_diarization_hardcases: benchmark_diarization_data
	uv run python $(BENCH_DIR)/prepare_hard_cases.py
	uv run python $(BENCH_DIR)/benchmark_baseline.py \
		--data $(BENCH_DIR)/data/hard_cases --suffix "" --stream-sec 0 \
		--skip-offline --label "$(LABEL)" \
		--out $(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_hardcases.json,$(RESULT))

# Long-session soak: 60 min (SOAK_MINUTES=120 for two hours) of concatenated
# AMI meetings, tracking drift, label swaps and memory growth per 5 min bin.
SOAK_MINUTES ?= 60
benchmark_diarization_soak:
	uv run python $(BENCH_DIR)/prepare_soak.py --minutes $(SOAK_MINUTES)
	uv run python $(BENCH_DIR)/soak.py --minutes $(SOAK_MINUTES) --label "$(LABEL)" \
		--out $(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_soak_$(SOAK_MINUTES)min.json,$(RESULT))

# Phase 2a acceptance: absolute targets (baselines/phase2a_acceptance.json)
# checked against a suite report; the Linux container run is the one that
# decides. RESULT=path picks the report (default: today's suite report).
benchmark_diarization_acceptance:
	uv run python $(BENCH_DIR)/acceptance.py --report $(RESULT)

# Concurrency: the suite plus N concurrent sessions with diarization on
# (SESSIONS="2 3 4"), reporting how many the environment sustains.
SESSIONS ?= 2 3 4
benchmark_diarization_concurrency: benchmark_diarization_data
	uv run python $(BENCH_DIR)/run_suite.py \
		--only caption concurrency --sessions-sweep $(SESSIONS) \
		--caption-seconds $(CAPTION_SEC) --label "$(LABEL)" \
		--out $(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_concurrency.json,$(RESULT)) $(SUITE_ARGS)

benchmark_diarization_concurrency_docker: benchmark_diarization_data
	$(BENCH_DIR)/docker/run_in_docker.sh \
		--only caption concurrency --sessions-sweep $(SESSIONS) \
		--caption-seconds $(CAPTION_SEC) --label "$(LABEL)" \
		--out $(subst $(BENCH_DIR)/,/app/$(BENCH_DIR)/,$(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_concurrency.json,$(RESULT))) $(SUITE_ARGS)

# Window / segmentation-step sweep (Phase 2a): the numbers the defaults were
# chosen from. WINDOWS="8 10 12 15 20" STEPS="0.1 0.25 0.5".
WINDOWS ?= 8 10 12 15 20
STEPS ?= 0.1 0.25 0.5
benchmark_diarization_window_sweep: benchmark_diarization_data
	uv run python $(BENCH_DIR)/window_sweep.py --windows $(WINDOWS) --steps $(STEPS) \
		--threads 1 --label "$(LABEL)" \
		--out $(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_window_sweep.json,$(RESULT))

benchmark_diarization_window_sweep_docker: benchmark_diarization_data
	BENCH_SCRIPT=window_sweep.py $(BENCH_DIR)/docker/run_in_docker.sh --windows $(WINDOWS) --steps $(STEPS) \
		--threads 1 --label "$(LABEL)" \
		--out $(subst $(BENCH_DIR)/,/app/$(BENCH_DIR)/,$(if $(filter $(BENCH_DIR)/results/$(DATE)_suite.json,$(RESULT)),$(BENCH_DIR)/results/$(DATE)_window_sweep.json,$(RESULT)))
