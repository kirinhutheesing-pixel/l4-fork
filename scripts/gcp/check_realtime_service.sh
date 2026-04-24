#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"
CHECK_FRAME="${CHECK_FRAME:-0}"
FRAME_PATH="${FRAME_PATH:-/tmp/falcon-pipeline-frame.jpg}"

health_file="$(mktemp)"
state_file="$(mktemp)"
trap 'rm -f "${health_file}" "${state_file}"' EXIT

nvidia-smi >/dev/null
sudo docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi >/dev/null

curl -fsS "http://127.0.0.1:${PORT}/api/healthz" > "${health_file}"
curl -fsS "http://127.0.0.1:${PORT}/api/state" > "${state_file}"

python3 - "${health_file}" "${state_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    health = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    state = json.load(handle)

if health.get("status") != "ok":
    raise SystemExit("healthz did not return status=ok")

readiness = state.get("readiness") or {}
source = state.get("source") or {}
blocking_engine_errors = readiness.get("blocking_engine_errors") or []
summary = {
    "service_state": readiness.get("service_state"),
    "integration_ready": readiness.get("integration_ready"),
    "full_pipeline_ready": readiness.get("full_pipeline_ready"),
    "sam3_visual_ready": readiness.get("sam3_visual_ready"),
    "error_kind": readiness.get("error_kind"),
    "error_message": readiness.get("error_message"),
    "blocking_engine_errors": blocking_engine_errors,
    "source_status": source.get("status"),
    "source_error_kind": source.get("error_kind"),
    "source_error_message": source.get("error_message"),
    "frame_ready": (state.get("frame") or {}).get("ready"),
}
print(json.dumps(summary, indent=2))

if readiness.get("full_pipeline_ready") is True:
    raise SystemExit(0)

if source.get("status") in {"auth_required", "unavailable"} and readiness.get("error_kind") in {"source_auth", "source_unavailable"}:
    raise SystemExit(0)

raise SystemExit(1)
PY

if [[ "${CHECK_FRAME}" == "1" ]]; then
  if python3 - "${state_file}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)
raise SystemExit(0 if (state.get("readiness") or {}).get("full_pipeline_ready") is True else 1)
PY
  then
    curl -fsS "http://127.0.0.1:${PORT}/api/frame.jpg" -o "${FRAME_PATH}"
    if [[ ! -s "${FRAME_PATH}" ]]; then
      echo "frame.jpg fetch returned an empty file" >&2
      exit 1
    fi
    echo "frame_path=${FRAME_PATH}"
  fi
fi
