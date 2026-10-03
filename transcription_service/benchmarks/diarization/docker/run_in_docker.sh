#!/usr/bin/env bash
# Run a diarization benchmark script inside the Linux CPU reference image
# with fixed resource limits.
#
# Usage (from transcription_service/):
#   benchmarks/diarization/docker/run_in_docker.sh [run_suite.py args...]
#   BENCH_CPUS=2 BENCH_MEMORY_GB=4 benchmarks/diarization/docker/run_in_docker.sh ...
#   BENCH_SCRIPT=caption_latency.py benchmarks/diarization/docker/run_in_docker.sh --help
#
# Environment:
#   BENCH_CPUS       CPU limit passed to docker --cpus          (default 4)
#   BENCH_MEMORY_GB  memory limit passed to docker --memory     (default 8)
#   BENCH_SCRIPT     script under benchmarks/diarization/       (default run_suite.py)
#   (both images are always built; the layer cache makes an unchanged build fast)
#   HUGGINGFACE_ACCESS_TOKEN or HF_TOKEN  gated pyannote model access
#
# Audio data, results and the HuggingFace / torch hub caches are bind-mounted
# from the host, so the same downloaded models and AMI files serve both the
# native and the container runs. Paths passed to the script must be the
# in-container ones (/app/benchmarks/diarization/...); the defaults already
# are.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/.." && pwd)"
SERVICE="$(cd "$BENCH/../.." && pwd)"

CPUS="${BENCH_CPUS:-4}"
MEMORY_GB="${BENCH_MEMORY_GB:-8}"
SCRIPT="${BENCH_SCRIPT:-run_suite.py}"
BASE_IMAGE="scribear-transcription-cpu:bench"
BENCH_IMAGE="scribear-diarization-bench:cpu"
TOKEN="${HUGGINGFACE_ACCESS_TOKEN:-${HF_TOKEN:-}}"

if [ -z "$TOKEN" ]; then
  echo "HUGGINGFACE_ACCESS_TOKEN (or HF_TOKEN) must be set for the pyannote model" >&2
  exit 1
fi
command -v docker >/dev/null || { echo "docker is required" >&2; exit 1; }
docker info >/dev/null 2>&1 || { echo "docker daemon is not running" >&2; exit 1; }

# Docker Desktop runs containers in a VM; a memory limit above the VM's RAM
# is only nominal. Say so rather than silently reporting an 8 GB limit that
# a 6 GB VM cannot honour.
VM_MEM_BYTES="$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)"
VM_MEM_GB=$(( VM_MEM_BYTES / 1024 / 1024 / 1024 ))
if [ "$VM_MEM_BYTES" -gt 0 ] && [ "$VM_MEM_GB" -lt "$MEMORY_GB" ]; then
  echo "WARNING: docker VM has ${VM_MEM_GB} GB; the ${MEMORY_GB} GB limit cannot be honoured" >&2
fi
VM_CPUS="$(docker info --format '{{.NCPU}}' 2>/dev/null || echo 0)"
if [ "$VM_CPUS" -gt 0 ] && [ "$VM_CPUS" -lt "$CPUS" ]; then
  echo "WARNING: docker VM has ${VM_CPUS} CPUs; the ${CPUS} CPU limit cannot be honoured" >&2
fi

# Always built: the base image carries the service's `src`, and a benchmark
# that silently ran yesterday's service against today's config is a wasted
# hour (it happened). Docker's layer cache makes an unchanged rebuild take
# seconds; only a changed `src` re-runs the final `uv sync`.
echo "--- building $BASE_IMAGE from Dockerfile_CPU (upstream's production CPU image)"
docker build -f "$SERVICE/Dockerfile_CPU" -t "$BASE_IMAGE" "$SERVICE"
echo "--- building $BENCH_IMAGE (base + pyannote extra + benchmark sources)"
docker build -f "$HERE/Dockerfile" --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$BENCH_IMAGE" "$SERVICE"

mkdir -p "$BENCH/data" "$BENCH/results" "$HOME/.cache/huggingface" "$HOME/.cache/torch"

echo "--- running $SCRIPT with --cpus $CPUS --memory ${MEMORY_GB}g"
exec docker run --rm -i \
  --cpus "$CPUS" --memory "${MEMORY_GB}g" --memory-swap "${MEMORY_GB}g" \
  -e HUGGINGFACE_ACCESS_TOKEN="$TOKEN" \
  -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e BENCH_CPU_LIMIT="$CPUS" -e BENCH_MEM_LIMIT_GB="$MEMORY_GB" \
  -e SCRIBEAR_BENCH_RUNNER=docker \
  -e SCRIBEAR_BUILD_COMMIT="$(git -C "$SERVICE" rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  -v "$BENCH/data:/app/benchmarks/diarization/data" \
  -v "$BENCH/results:/app/benchmarks/diarization/results" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/torch:/root/.cache/torch" \
  "$BENCH_IMAGE" "benchmarks/diarization/$SCRIPT" "$@"
