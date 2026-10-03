# Linux CPU reference environment for the diarization benchmark.
#
# Production runs the CPU image built from transcription_service/Dockerfile_CPU
# on Linux. This image is that exact image plus the pyannote extra (the
# production image does not ship it yet) and the benchmark sources, so every
# number produced inside it reflects upstream's torch/OpenBLAS/thread settings
# (OMP_NUM_THREADS=1, Silero's torch.set_num_threads(1), ...) rather than a
# developer laptop.
#
# Build (from transcription_service/):
#   docker build -f Dockerfile_CPU -t scribear-transcription-cpu:bench .
#   docker build -f benchmarks/diarization/docker/Dockerfile \
#       -t scribear-diarization-bench:cpu .
# or simply run benchmarks/diarization/docker/run_in_docker.sh, which does both.
ARG BASE_IMAGE=scribear-transcription-cpu:bench
FROM ${BASE_IMAGE}

# ffmpeg: shared libraries torchcodec (a pyannote.audio 4 dependency) loads at
# import time. Audio preparation itself happens on the host.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Same extras as Dockerfile_CPU plus pyannote, from the same frozen lockfile.
RUN uv sync --frozen --no-editable \
    --extra faster-whisper \
    --extra silero-vad-cpu \
    --extra pyannote-diarization

COPY benchmarks ./benchmarks
COPY tests ./tests

# Benchmark scripts import `src.*` from the repo root, as the service does.
ENV PYTHONPATH=/app
ENV SCRIBEAR_BENCH_RUNNER=docker

ENTRYPOINT ["python"]
CMD ["benchmarks/diarization/run_suite.py", "--help"]
