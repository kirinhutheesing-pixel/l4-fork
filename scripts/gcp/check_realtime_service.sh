#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"
CHECK_FRAME="${CHECK_FRAME:-0}"
CHECK_RESTAURANT_CONTRACT="${CHECK_RESTAURANT_CONTRACT:-0}"
MAX_SAM3_AGE_SECONDS="${MAX_SAM3_AGE_SECONDS:-10}"
FRAME_PATH="${FRAME_PATH:-/tmp/falcon-pipeline-frame.jpg}"
export CHECK_RESTAURANT_CONTRACT MAX_SAM3_AGE_SECONDS

health_file="$(mktemp)"
state_file="$(mktemp)"
trap 'rm -f "${health_file}" "${state_file}"' EXIT

nvidia-smi >/dev/null
sudo docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 nvidia-smi >/dev/null

curl -fsS "http://127.0.0.1:${PORT}/api/healthz" > "${health_file}"
curl -fsS "http://127.0.0.1:${PORT}/api/state" > "${state_file}"

python3 - "${health_file}" "${state_file}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    health = json.load(handle)
with open(sys.argv[2], encoding="utf-8") as handle:
    state = json.load(handle)

if health.get("status") != "ok":
    raise SystemExit("healthz did not return status=ok")

readiness = state.get("readiness") or {}
source = state.get("source") or {}
result = state.get("result") or {}
metrics = state.get("metrics") or {}
blocking_engine_errors = readiness.get("blocking_engine_errors") or []
engines = result.get("engines") or []
sam3_engine = next(
    (
        engine
        for engine in engines
        if str(engine.get("name") or "").replace("-", "_").lower() == "sam3"
    ),
    {},
)
sam3_num_masks = result.get("num_masks") or 0
sam3_age_seconds = result.get("sam3_segmentation_age_seconds")
if sam3_age_seconds is None:
    sam3_age_seconds = metrics.get("sam3_segmentation_age_seconds")
sam3_segmentation_state = metrics.get("sam3_segmentation_state")

summary = {
    "service_state": readiness.get("service_state"),
    "integration_ready": readiness.get("integration_ready"),
    "full_pipeline_ready": readiness.get("full_pipeline_ready"),
    "sam3_visual_ready": readiness.get("sam3_visual_ready"),
    "primary_engine": result.get("primary_engine"),
    "sam3_engine_status": sam3_engine.get("status"),
    "sam3_num_masks": sam3_num_masks,
    "sam3_segmentation_state": sam3_segmentation_state,
    "sam3_age_seconds": sam3_age_seconds,
    "error_kind": readiness.get("error_kind"),
    "error_message": readiness.get("error_message"),
    "blocking_engine_errors": blocking_engine_errors,
    "source_status": source.get("status"),
    "source_error_kind": source.get("error_kind"),
    "source_error_message": source.get("error_message"),
    "frame_ready": (state.get("frame") or {}).get("ready"),
}
print(json.dumps(summary, indent=2))

if source.get("status") in {"auth_required", "unavailable"} and readiness.get("error_kind") in {
    "source_auth",
    "source_unavailable",
}:
    raise SystemExit(0)

def fail(message: str) -> None:
    raise SystemExit(message)

if readiness.get("full_pipeline_ready") is not True:
    fail("full_pipeline_ready is not true")
if readiness.get("sam3_visual_ready") is not True:
    fail("sam3_visual_ready is not true")
if result.get("primary_engine") != "sam3":
    fail("primary_engine is not sam3")
if sam3_engine.get("status") != "ok":
    fail("SAM3 engine status is not ok")
if not isinstance(sam3_num_masks, int) or sam3_num_masks <= 0:
    fail("SAM3 did not report any masks")
if sam3_segmentation_state not in {"ready", "segmenting"}:
    fail("SAM3 segmentation state is not ready or segmenting")

if sam3_segmentation_state == "segmenting":
    max_age_seconds = float(os.environ.get("MAX_SAM3_AGE_SECONDS", "10"))
    try:
        age_value = float(sam3_age_seconds)
    except (TypeError, ValueError):
        fail("SAM3 is segmenting but no prior result age is available")
    if age_value > max_age_seconds:
        fail(f"SAM3 is segmenting but the latest result is stale: {age_value:.3f}s")

if os.environ.get("CHECK_RESTAURANT_CONTRACT") == "1":
    scene = result.get("scene_annotations") or {}
    if scene.get("profile") != "restaurant_service":
        fail("restaurant scene_annotations.profile is not restaurant_service")
    entities = scene.get("entities")
    if not isinstance(entities, list) or not entities:
        fail("restaurant scene_annotations.entities is missing or empty")
    required_person_fields = {
        "role_reason",
        "role_confidence",
        "classification_source",
        "near_guest_context",
    }
    for entity in entities:
        if entity.get("kind") != "person":
            continue
        missing = sorted(required_person_fields - set(entity))
        if missing:
            fail(f"restaurant person entity is missing fields: {', '.join(missing)}")

raise SystemExit(0)
PY

if [[ "${CHECK_FRAME}" == "1" ]]; then
  if python3 - "${state_file}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)
readiness = state.get("readiness") or {}
result = state.get("result") or {}
raise SystemExit(
    0
    if readiness.get("full_pipeline_ready") is True
    and readiness.get("sam3_visual_ready") is True
    and result.get("primary_engine") == "sam3"
    else 1
)
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
