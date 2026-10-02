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

# Speaker diarization accuracy/speed baseline on AMI meetings (CPU).
# First run downloads ~110 MB of AMI audio into benchmarks/diarization/data
# (gitignored). Needs HUGGINGFACE_ACCESS_TOKEN or HF_TOKEN for the gated
# pyannote model. Override RESULT=... to write a differently named report.
RESULT ?= benchmarks/diarization/results/$(shell date +%Y%m%d)_baseline.json
STREAM_SEC ?= 120
benchmark_diarization_baseline:
	benchmarks/diarization/prepare_ami_baseline.sh 10
	uv run python benchmarks/diarization/benchmark_baseline.py \
		--data benchmarks/diarization/data/ami --suffix _10min \
		--stream-sec $(STREAM_SEC) --out $(RESULT)
