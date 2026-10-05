#!/usr/bin/env bash
# Start-up test for a diarization image with networking disabled.
#
# Proves the three things the image promises: it starts with no network and
# no HuggingFace token (the model is baked in), the diarization worker loads
# the model warm within the time budget, and a bad diarization config fails
# at start-up with a readable message instead of a per-session reconnect loop.
#
# Usage (from transcription_service/):
#   scripts/check_diarization_image.sh scribear/transcription-service-cpu-diarization:dev
#
# Environment:
#   WHISPER_CACHE   host directory mounted at /models/hf (HF_HOME in compose),
#                   which must already hold the whisper model the config names
#                   (default ~/.cache/huggingface). With the network off the
#                   service cannot download anything, which is the point.
#   TORCH_CACHE     host torch hub cache holding Silero VAD (default ~/.cache/torch)
#   READY_TIMEOUT   seconds to wait for readiness (default 180)
#   WARM_LOAD_BUDGET_SEC  the model-load budget asserted from the log (default 10)
set -euo pipefail

IMAGE="${1:?image tag required}"
WHISPER_CACHE="${WHISPER_CACHE:-$HOME/.cache/huggingface}"
TORCH_CACHE="${TORCH_CACHE:-$HOME/.cache/torch}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"
WARM_LOAD_BUDGET_SEC="${WARM_LOAD_BUDGET_SEC:-10}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="$ROOT/deployment/provider_config.diarization.template.json"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

run_service() {
  # $1 container name, $2 provider config path. No network, no token.
  docker run -d --rm --name "$1" --network none \
    -e API_KEY=startup-test -e METRICS_API_KEY=startup-test -e WS_INIT_TIMEOUT_SEC=2.5 \
    -e LOG_LEVEL=info \
    -e HF_HOME=/models/hf \
    -v "$2:/app/provider_config.json:ro" \
    -v "$WHISPER_CACHE:/models/hf" \
    -v "$TORCH_CACHE:/root/.cache/torch" \
    "$IMAGE" >/dev/null
}

wait_ready() {
  local name="$1" deadline=$((SECONDS + READY_TIMEOUT))
  while [ $SECONDS -lt $deadline ]; do
    if ! docker ps -q --filter "name=^${name}$" | grep -q .; then
      echo "FAIL: container exited before readiness"; docker logs "$name" 2>&1 | tail -20; return 1
    fi
    if docker exec "$name" curl -fs http://localhost:80/probes/readiness >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "FAIL: not ready after ${READY_TIMEOUT}s"; docker logs "$name" 2>&1 | tail -30; return 1
}

echo "== 1. offline start: no network, no token, baked model =="
docker rm -f diar-startup-ok >/dev/null 2>&1 || true
run_service diar-startup-ok "$CONFIG"
START=$SECONDS
wait_ready diar-startup-ok
echo "ready after $((SECONDS - START))s"
HEALTH="$(docker exec diar-startup-ok curl -fs -H 'Authorization: Bearer startup-test' http://localhost:80/providers/health || true)"
echo "providers/health: ${HEALTH:0:400}"
echo "$HEALTH" | grep -q '"whisper"' || { echo "FAIL: whisper provider missing from health"; exit 1; }
echo "$HEALTH" | grep -qi '"status": *"ok"\|"status":"ok"\|"OK"' || { echo "FAIL: provider not OK: $HEALTH"; exit 1; }
LOGS="$(docker logs diar-startup-ok 2>&1)"
echo "$LOGS" | grep -q "diarization model loaded successfully" || { echo "FAIL: model load line missing"; echo "$LOGS" | tail -30; exit 1; }
echo "$LOGS" | grep -q "from local_dir" || { echo "FAIL: model was not loaded from the baked directory"; exit 1; }
LOAD_SEC="$(echo "$LOGS" | grep -oE "loaded successfully in [0-9.]+s" | head -1 | grep -oE "[0-9]+(\.[0-9]+)?" | head -1)"
echo "diarization model warm load: ${LOAD_SEC}s (budget ${WARM_LOAD_BUDGET_SEC}s)"
awk -v l="$LOAD_SEC" -v b="$WARM_LOAD_BUDGET_SEC" 'BEGIN { exit !(l+0 <= b+0) }' || { echo "FAIL: warm load over budget"; exit 1; }
METRICS="$(docker exec diar-startup-ok curl -fs -H 'Authorization: Bearer startup-test' http://localhost:80/metrics/status || true)"
echo "$METRICS" | grep -q '"deviceFallbacks"' || { echo "FAIL: metrics lack deviceFallbacks"; exit 1; }
echo "metrics/status providerDevice: $(echo "$METRICS" | grep -o '"providerDevice": *{[^}]*}' | head -1)"
docker stop diar-startup-ok >/dev/null
echo "PASS"

echo
echo "== 2. bad diarization tag fails at start-up with a message =="
sed 's/"diarization_context_tag": "pyannote_diarization"/"diarization_context_tag": "no_such_context"/' "$CONFIG" > "$SCRATCH/bad_tag.json"
docker rm -f diar-startup-bad >/dev/null 2>&1 || true
set +e
docker run --rm --name diar-startup-bad --network none \
  -e API_KEY=startup-test -e WS_INIT_TIMEOUT_SEC=2.5 -e HF_HOME=/models/hf \
  -v "$SCRATCH/bad_tag.json:/app/provider_config.json:ro" \
  -v "$WHISPER_CACHE:/models/hf" -v "$TORCH_CACHE:/root/.cache/torch" \
  "$IMAGE" > "$SCRATCH/bad_tag.log" 2>&1
CODE=$?
set -e
echo "exit code $CODE"
grep -o "no live worker owns a context tagged 'no_such_context'[^\"]*" "$SCRATCH/bad_tag.log" | head -1 || { echo "FAIL: message missing"; tail -20 "$SCRATCH/bad_tag.log"; exit 1; }
[ "$CODE" -ne 0 ] || { echo "FAIL: service should not have started"; exit 1; }
echo "PASS"

echo
echo "== 3. missing model directory fails at start-up with a message =="
docker rm -f diar-startup-nomodel >/dev/null 2>&1 || true
set +e
docker run --rm --name diar-startup-nomodel --network none \
  -e API_KEY=startup-test -e WS_INIT_TIMEOUT_SEC=2.5 -e HF_HOME=/models/hf \
  -e SCRIBEAR_DIARIZATION_MODEL_DIR=/nonexistent \
  -v "$CONFIG:/app/provider_config.json:ro" \
  -v "$WHISPER_CACHE:/models/hf" -v "$TORCH_CACHE:/root/.cache/torch" \
  "$IMAGE" > "$SCRATCH/nomodel.log" 2>&1
CODE=$?
set -e
echo "exit code $CODE"
grep -o "SCRIBEAR_DIARIZATION_MODEL_DIR=[^\"]*holds no config.yaml[^\"]*" "$SCRATCH/nomodel.log" | head -1 || { echo "FAIL: message missing"; tail -20 "$SCRATCH/nomodel.log"; exit 1; }
[ "$CODE" -ne 0 ] || { echo "FAIL: service should not have started"; exit 1; }
echo "PASS"
echo
echo "all start-up checks passed for $IMAGE"
