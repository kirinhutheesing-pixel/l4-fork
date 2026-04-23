#!/usr/bin/env bash
set -euo pipefail

: "${TEST_SOURCE_URL:?TEST_SOURCE_URL is required}"
: "${TEST_PROMPT:?TEST_PROMPT is required}"

IMAGE_NAME="${IMAGE_NAME:-falcon-pipeline:l4}"
CONTAINER_NAME="${CONTAINER_NAME:-falcon-pipeline-l4}"
PORT="${PORT:-8080}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/opt/falcon-pipeline/hf-cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/opt/falcon-pipeline/outputs}"
COOKIE_TARGET="/run/secrets/youtube-cookies.txt"

mkdir -p "${HF_CACHE_DIR}" "${OUTPUT_DIR}"

common_args=(
  -v "${HF_CACHE_DIR}:/app/.cache/huggingface"
  -v "${OUTPUT_DIR}:/app/outputs/falcon-pipeline-realtime"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
  common_args+=(
    -e "HF_TOKEN=${HF_TOKEN}"
    -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}"
  )
fi

cookie_args=()
service_cookie_args=()
if [[ -n "${YTDLP_COOKIES_FILE:-}" ]]; then
  if [[ ! -f "${YTDLP_COOKIES_FILE}" ]]; then
    echo "Cookie file not found: ${YTDLP_COOKIES_FILE}" >&2
    exit 21
  fi
  cookie_args+=(-v "${YTDLP_COOKIES_FILE}:${COOKIE_TARGET}:ro")
  service_cookie_args+=(--yt-cookies-file "${COOKIE_TARGET}")
fi

service_entrypoint=(
  --entrypoint /app/.venv/bin/python
  "${IMAGE_NAME}"
  /app/falcon_pipeline_realtime_service.py
)

preflight_cmd=(
  sudo docker run --rm
  "${common_args[@]}"
  "${cookie_args[@]}"
  "${service_entrypoint[@]}"
  --source-url "${TEST_SOURCE_URL}"
  --preflight-only
  "${service_cookie_args[@]}"
)

set +e
preflight_output="$("${preflight_cmd[@]}" 2>&1)"
preflight_exit=$?
set -e

echo "${preflight_output}"

if [[ ${preflight_exit} -ne 0 ]]; then
  exit "${preflight_exit}"
fi

sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

launch_cmd=(
  sudo docker run -d
  --name "${CONTAINER_NAME}"
  --restart unless-stopped
  --gpus all
  -p "${PORT}:8080"
  "${common_args[@]}"
  "${cookie_args[@]}"
  "${service_entrypoint[@]}"
  --source-url "${TEST_SOURCE_URL}"
  --prompt "${TEST_PROMPT}"
  --cache-dir /app/.cache/huggingface
  --output-dir /app/outputs/falcon-pipeline-realtime
  --no-compile
  "${service_cookie_args[@]}"
)

container_id="$("${launch_cmd[@]}")"

echo "Started container ${CONTAINER_NAME}: ${container_id}"
echo "Port: ${PORT}"
echo "Next: bash scripts/gcp/check_realtime_service.sh"
