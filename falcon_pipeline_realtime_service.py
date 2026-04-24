#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from PIL import Image

ROOT = Path(__file__).resolve().parent
FALCON_SOURCE_ROOT = ROOT / "Falcon-Perception-main"
if FALCON_SOURCE_ROOT.exists() and str(FALCON_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(FALCON_SOURCE_ROOT))

from falcon_perception import (  # noqa: E402
    PERCEPTION_300M_MODEL_ID,
    build_prompt_for_task,
    load_and_prepare_model,
    setup_torch_config,
)

from perception_orchestrator import (  # noqa: E402
    DEFAULT_RT_DETR_MODEL_ID,
    DEFAULT_SAM3_MODEL_ID,
    ModelAccessError,
    ModelUnsupportedError,
    build_orchestrated_inference,
    classify_engine_error_kind,
    huggingface_token,
    inspect_torch_stack,
    intersection_over_union,
    load_rtdetr_runtime,
    load_sam3_runtime,
    run_rtdetr_inference,
    run_sam3_inference,
)
from run_falcon_pipeline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    configure_huggingface_cache,
    is_youtube_url,
    make_slug,
    open_video_capture,
    pair_bbox_entries,
    render_visualization,
    resolve_stream_source,
    summarize_decoded_output,
)

DEFAULT_UI_PATH = ROOT / "falcon_pipeline_realtime_ui.html"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "falcon-pipeline-realtime"

PROMPT_GUIDELINES = [
    {
        "title": "Use referring expressions, not open-ended questions.",
        "good": "people under the blue umbrellas",
        "avoid": "what is happening here?",
    },
    {
        "title": "Name visible objects and attributes.",
        "good": "white boats on the horizon",
        "avoid": "interesting things near the back",
    },
    {
        "title": "Add position when there are many similar objects.",
        "good": "tables on the left side of the deck",
        "avoid": "tables",
    },
    {
        "title": "Keep each live prompt focused on one target concept.",
        "good": "people standing near the railing",
        "avoid": "people, umbrellas, tables, and clouds",
    },
]

RESTAURANT_ROLE_COLORS = {
    "restaurant_goer": (36, 201, 138),
    "server": (71, 144, 255),
    "table": (88, 197, 255),
    "unclassified": (168, 184, 196),
    "needs_service": (54, 54, 235),
}

SOURCE_STATUS_IDLE = "idle"
SOURCE_STATUS_RESOLVING = "resolving"
SOURCE_STATUS_READY = "ready"
SOURCE_STATUS_AUTH_REQUIRED = "auth_required"
SOURCE_STATUS_UNAVAILABLE = "unavailable"

ERROR_KIND_SOURCE_AUTH = "source_auth"
ERROR_KIND_SOURCE_UNAVAILABLE = "source_unavailable"
ERROR_KIND_MODEL_LOAD = "model_load"
ERROR_KIND_MODEL_ACCESS = "model_access"
ERROR_KIND_MODEL_UNSUPPORTED = "model_unsupported"
ERROR_KIND_INFERENCE_RUNTIME = "inference_runtime"

SAM_PREFLIGHT_STATUS_READY = "ready"
SAM_PREFLIGHT_STATUS_MISSING_TOKEN = "missing_token"
SAM_PREFLIGHT_STATUS_ACCESS_REQUIRED = "access_required"
SAM_PREFLIGHT_STATUS_UNSUPPORTED = "unsupported"
SAM_PREFLIGHT_STATUS_LOAD_FAILED = "load_failed"


def resize_image_to_bounds(image: Image.Image, *, min_dim: int, max_dim: int) -> Image.Image:
    width, height = image.size
    shortest = min(width, height)
    longest = max(width, height)
    scale = 1.0

    if longest > max_dim:
        scale = max_dim / float(longest)
    elif shortest < min_dim:
        scale = min_dim / float(shortest)

    if abs(scale - 1.0) < 1e-3:
        return image

    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.BICUBIC)


def detection_bbox_xyxy(detection: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    center = detection.get("center") or {}
    cx = float(center.get("x", 0.5)) * width
    cy = float(center.get("y", 0.5)) * height
    bw = float(detection.get("width", 0.0)) * width
    bh = float(detection.get("height", 0.0)) * height
    x0 = int(round(cx - bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    x1 = int(round(cx + bw / 2.0))
    y1 = int(round(cy + bh / 2.0))
    return x0, y0, x1, y1


def bbox_norm_xyxy(bbox: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    cx = float(bbox.get("x", 0.5)) * width
    cy = float(bbox.get("y", 0.5)) * height
    bw = float(bbox.get("w", 0.0)) * width
    bh = float(bbox.get("h", 0.0)) * height
    x0 = int(round(cx - bw / 2.0))
    y0 = int(round(cy - bh / 2.0))
    x1 = int(round(cx + bw / 2.0))
    y1 = int(round(cy + bh / 2.0))
    return x0, y0, x1, y1


def detection_bbox_norm(detection: dict[str, Any]) -> dict[str, float]:
    center = detection.get("center") or {}
    return {
        "x": float(center.get("x", 0.5)),
        "y": float(center.get("y", 0.5)),
        "w": float(detection.get("width", 0.0)),
        "h": float(detection.get("height", 0.0)),
    }


def detection_overlaps(detection: dict[str, Any], candidates: list[dict[str, Any]], threshold: float = 0.2) -> bool:
    bbox = detection_bbox_norm(detection)
    for candidate in candidates:
        if intersection_over_union(bbox, detection_bbox_norm(candidate)) >= threshold:
            return True
    return False


def detection_area_norm(detection: dict[str, Any]) -> float:
    bbox = detection_bbox_norm(detection)
    return max(0.0, bbox["w"]) * max(0.0, bbox["h"])


def is_broad_guidance_detection(detection: dict[str, Any]) -> bool:
    bbox = detection_bbox_norm(detection)
    return detection_area_norm(detection) >= 0.60 or (bbox["w"] >= 0.85 and bbox["h"] >= 0.85)


def focused_guidance_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [detection for detection in detections if not is_broad_guidance_detection(detection)]


def first_guidance_overlap(
    person: dict[str, Any],
    guidance_sources: list[tuple[str, list[dict[str, Any]]]],
    *,
    threshold: float = 0.18,
) -> str | None:
    for source_name, detections in guidance_sources:
        if detection_overlaps(person, focused_guidance_detections(detections), threshold=threshold):
            return source_name
    return None


def is_restaurant_tracking_prompt(prompt: str) -> bool:
    lowered = prompt.lower()
    markers = ("restaurant", "table", "server", "service", "guest", "diner", "waiter", "waitress")
    return any(marker in lowered for marker in markers)


def entity_color_bgr(entity: dict[str, Any]) -> tuple[int, int, int]:
    if entity.get("needs_service"):
        return RESTAURANT_ROLE_COLORS["needs_service"]
    role = str(entity.get("role") or "")
    if role == "server":
        return RESTAURANT_ROLE_COLORS["server"]
    if role == "table":
        return RESTAURANT_ROLE_COLORS["table"]
    if role == "unclassified":
        return RESTAURANT_ROLE_COLORS["unclassified"]
    return RESTAURANT_ROLE_COLORS["restaurant_goer"]


def infer_source_type(source_url: str) -> str | None:
    if not source_url.strip():
        return None
    return "youtube" if is_youtube_url(source_url) else "direct"


def classify_source_error(message: str) -> tuple[str, str]:
    lowered = message.lower()
    auth_markers = (
        "sign in to confirm you",
        "--cookies-from-browser",
        "use --cookies",
        "for the authentication",
        "cookie",
    )
    if any(marker in lowered for marker in auth_markers):
        return SOURCE_STATUS_AUTH_REQUIRED, ERROR_KIND_SOURCE_AUTH
    return SOURCE_STATUS_UNAVAILABLE, ERROR_KIND_SOURCE_UNAVAILABLE


def classify_model_load_error_kind(message: str | None) -> str:
    engine_error_kind = classify_engine_error_kind(message)
    if engine_error_kind in {ERROR_KIND_MODEL_ACCESS, ERROR_KIND_MODEL_UNSUPPORTED}:
        return engine_error_kind
    return ERROR_KIND_MODEL_LOAD


def build_source_state(
    *,
    input_url: str,
    source_type: str | None,
    status: str,
    cookie_file_configured: bool,
    resolved_url_present: bool = False,
    title: str | None = None,
    channel: str | None = None,
    is_live: bool | None = None,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "input_url": input_url,
        "source_type": source_type,
        "status": status,
        "resolved_url_present": resolved_url_present,
        "title": title,
        "channel": channel,
        "is_live": is_live,
        "cookie_file_configured": cookie_file_configured,
        "error_kind": error_kind,
        "error_message": error_message,
    }


def build_source_state_from_stream_info(
    *,
    input_url: str,
    cookie_file: Path | None,
    stream_info: dict[str, Any] | None,
    status: str,
    error_kind: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return build_source_state(
        input_url=input_url,
        source_type=(stream_info or {}).get("source_type") or infer_source_type(input_url),
        status=status,
        resolved_url_present=bool((stream_info or {}).get("resolved_url")),
        title=(stream_info or {}).get("title"),
        channel=(stream_info or {}).get("channel"),
        is_live=(stream_info or {}).get("is_live"),
        cookie_file_configured=cookie_file is not None,
        error_kind=error_kind,
        error_message=error_message,
    )


def probe_source(
    source_url: str,
    *,
    cookie_file: Path | None,
    timeout_seconds: float,
    keep_capture_open: bool = False,
) -> tuple[dict[str, Any], int, dict[str, Any] | None, cv2.VideoCapture | None]:
    trimmed_source_url = source_url.strip()
    if not trimmed_source_url:
        state = build_source_state(
            input_url="",
            source_type=None,
            status=SOURCE_STATUS_UNAVAILABLE,
            cookie_file_configured=cookie_file is not None,
            error_kind=ERROR_KIND_SOURCE_UNAVAILABLE,
            error_message="No source URL provided.",
        )
        return state, 21, None, None

    stream_info: dict[str, Any] | None = None
    try:
        stream_info = resolve_stream_source(trimmed_source_url, cookie_file=cookie_file)
    except (Exception, SystemExit) as exc:
        error_message = str(exc)
        status, error_kind = classify_source_error(error_message)
        state = build_source_state_from_stream_info(
            input_url=trimmed_source_url,
            cookie_file=cookie_file,
            stream_info=stream_info,
            status=status,
            error_kind=error_kind,
            error_message=error_message,
        )
        return state, (20 if error_kind == ERROR_KIND_SOURCE_AUTH else 21), stream_info, None

    try:
        capture = open_video_capture(stream_info, timeout_seconds)
    except (Exception, SystemExit) as exc:
        error_message = str(exc)
        status, error_kind = classify_source_error(error_message)
        state = build_source_state_from_stream_info(
            input_url=trimmed_source_url,
            cookie_file=cookie_file,
            stream_info=stream_info,
            status=status,
            error_kind=error_kind,
            error_message=error_message,
        )
        return state, (20 if error_kind == ERROR_KIND_SOURCE_AUTH else 21), stream_info, None

    state = build_source_state_from_stream_info(
        input_url=trimmed_source_url,
        cookie_file=cookie_file,
        stream_info=stream_info,
        status=SOURCE_STATUS_READY,
    )
    if not keep_capture_open:
        capture.release()
        capture = None
    return state, 0, stream_info, capture


def person_is_near_anchor(person: dict[str, Any], anchors: list[dict[str, Any]]) -> bool:
    person_bbox = detection_bbox_norm(person)
    px = person_bbox["x"]
    py = person_bbox["y"]
    pw = person_bbox["w"]
    ph = person_bbox["h"]
    for anchor in anchors:
        anchor_bbox = detection_bbox_norm(anchor)
        tx = anchor_bbox["x"]
        ty = anchor_bbox["y"]
        tw = anchor_bbox["w"]
        th = anchor_bbox["h"]
        horizontal_close = abs(px - tx) <= max(tw * 0.9, pw * 1.25)
        vertical_close = py >= ty - (th * 1.4 + ph * 0.4) and py <= ty + (th * 1.3)
        if horizontal_close and vertical_close:
            return True
        if intersection_over_union(person_bbox, anchor_bbox) >= 0.02:
            return True
    return False


def person_is_near_table(person: dict[str, Any], tables: list[dict[str, Any]]) -> bool:
    return person_is_near_anchor(person, tables)


def is_guest_context_detection(detection: dict[str, Any]) -> bool:
    label = str(detection.get("label") or "").lower()
    markers = (
        "dining table",
        "table",
        "chair",
        "bench",
        "cup",
        "bottle",
        "wine glass",
        "plate",
        "bowl",
    )
    return any(marker in label for marker in markers)


def build_restaurant_scene_annotations(inference: dict[str, Any], prompt: str) -> dict[str, Any] | None:
    if not is_restaurant_tracking_prompt(prompt):
        return None

    engine_outputs = inference.get("engine_outputs") or {}
    rtdetr_output = engine_outputs.get("rt_detr") or {}
    falcon_output = engine_outputs.get("falcon") or {}
    sam3_output = engine_outputs.get("sam3") or {}
    candidate_detections = rtdetr_output.get("candidate_detections") or []

    people = [det for det in candidate_detections if str(det.get("label") or "").lower() == "person"]
    tables = [det for det in candidate_detections if "table" in str(det.get("label") or "").lower()]
    guest_context = [det for det in candidate_detections if is_guest_context_detection(det)]
    guidance_sources = [
        ("falcon", falcon_output.get("detections") or []),
        ("sam3", sam3_output.get("detections") or []),
    ]
    service_prompt = "service" in prompt.lower()

    entities: list[dict[str, Any]] = []
    goer_count = 0
    server_count = 0
    unclassified_count = 0
    needs_service_count = 0

    for index, table in enumerate(tables, start=1):
        entities.append(
            {
                "entity_id": f"table-{index}",
                "kind": "table",
                "role": "table",
                "needs_service": False,
                "bbox": detection_bbox_norm(table),
            }
        )

    for index, person in enumerate(people, start=1):
        near_table = person_is_near_table(person, tables)
        near_guest_context = person_is_near_anchor(person, guest_context)
        guidance_source = first_guidance_overlap(person, guidance_sources, threshold=0.18)
        needs_service = bool(service_prompt and guidance_source)
        if needs_service:
            role = "restaurant_goer"
            role_reason = "focused_service_guidance"
            role_confidence = 0.85
            classification_source = guidance_source or "falcon"
            needs_service_count += 1
            goer_count += 1
        elif near_table:
            role = "restaurant_goer"
            role_reason = "near_table"
            role_confidence = 0.70
            classification_source = "rt_detr"
            goer_count += 1
        elif near_guest_context:
            role = "restaurant_goer"
            role_reason = "near_guest_context"
            role_confidence = 0.55
            classification_source = "rt_detr"
            goer_count += 1
        elif tables:
            role = "server"
            role_reason = "away_from_guest_tables"
            role_confidence = 0.50
            classification_source = "heuristic"
            server_count += 1
        else:
            role = "unclassified"
            role_reason = "insufficient_grounding"
            role_confidence = 0.0
            classification_source = "none"
            unclassified_count += 1

        entities.append(
            {
                "entity_id": f"person-{index}",
                "kind": "person",
                "role": role,
                "needs_service": needs_service,
                "near_table": near_table,
                "near_guest_context": near_guest_context,
                "role_reason": role_reason,
                "role_confidence": role_confidence,
                "classification_source": classification_source,
                "bbox": detection_bbox_norm(person),
            }
        )

    return {
        "profile": "restaurant_service",
        "entities": entities,
        "counts": {
            "restaurant_goers": goer_count,
            "servers": server_count,
            "unclassified": unclassified_count,
            "tables": len(tables),
            "needs_service": needs_service_count,
        },
    }


def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(30, min(quality, 95)))],
    )
    if not ok:
        raise RuntimeError("Could not encode the overlay frame as JPEG.")
    return encoded.tobytes()


def placeholder_frame(message: str) -> bytes:
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (31, 44, 47)
    cv2.putText(
        canvas,
        "Falcon Pipeline // Realtime",
        (48, 96),
        cv2.FONT_HERSHEY_DUPLEX,
        1.5,
        (237, 215, 162),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message,
        (48, 164),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (220, 236, 233),
        2,
        cv2.LINE_AA,
    )
    return encode_jpeg(canvas)


def falcon_task_for_live_prompt(task: str) -> str:
    return "detection" if task in {"detection", "segmentation"} else task


@dataclass
class LiveSessionConfig:
    source_url: str = ""
    prompt: str = "people under the umbrellas"
    task: str = "segmentation"
    enable_sam3: bool = True
    min_dim: int = 480
    max_dim: int = 960
    falcon_min_dim: int = 256
    falcon_max_dim: int = 640
    falcon_max_new_tokens: int = 128
    falcon_temperature: float = 0.0
    falcon_refresh_seconds: float = 5.0
    rtdetr_threshold: float = 0.35
    sam3_threshold: float = 0.5
    sam3_mask_threshold: float = 0.5
    stream_open_timeout: float = 60.0
    stream_read_timeout: float = 20.0
    jpeg_quality: int = 85

    def __post_init__(self) -> None:
        self.task = "segmentation"
        self.enable_sam3 = True


@dataclass
class RuntimeOptions:
    cache_dir: Path
    output_dir: Path
    youtube_cookie_file: Path | None = None
    falcon_model_id: str = PERCEPTION_300M_MODEL_ID
    rtdetr_model_id: str = DEFAULT_RT_DETR_MODEL_ID
    sam3_model_id: str = DEFAULT_SAM3_MODEL_ID
    dtype: str = "bfloat16"
    device: str = "cuda"
    compile_model: bool = True
    load_rtdetr: bool = True
    load_sam3: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    ui_path: Path = DEFAULT_UI_PATH

    def __post_init__(self) -> None:
        self.load_sam3 = True


@dataclass
class FalconGuidance:
    inference: dict[str, Any]
    ran_at: float
    frame_id: int
    prompt_revision: int


@dataclass
class Sam3Guidance:
    inference: dict[str, Any]
    ran_at: float
    frame_id: int
    prompt_revision: int


@dataclass
class Sam3Request:
    image: Image.Image
    query: str
    prompt_bboxes: list[dict[str, float]]
    threshold: float
    mask_threshold: float
    frame_id: int
    prompt_revision: int


class FalconRealtimeRuntime:
    def __init__(self, options: RuntimeOptions):
        from falcon_perception.batch_inference import BatchInferenceEngine

        self.options = options
        setup_torch_config()
        self.model, self.tokenizer, self.model_args = load_and_prepare_model(
            hf_model_id=options.falcon_model_id,
            hf_local_dir=None,
            device=options.device,
            dtype=options.dtype,
            compile=options.compile_model,
            backend="torch",
        )
        self.engine = BatchInferenceEngine(self.model, self.tokenizer)

    def run(
        self,
        *,
        image: Image.Image,
        query: str,
        task: str,
        min_dim: int,
        max_dim: int,
        max_new_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        from falcon_perception.batch_inference import process_batch_and_generate

        prompt = build_prompt_for_task(query, task)
        batch = process_batch_and_generate(
            self.tokenizer,
            [(image, prompt)],
            max_length=self.model_args.max_seq_len,
            min_dimension=min_dim,
            max_dimension=max_dim,
        )

        device = self.model.device
        tokens = batch["tokens"].to(device=device)
        pos_t = batch["pos_t"].to(device=device)
        pos_hw = batch["pos_hw"].to(device=device)
        pixel_values = batch["pixel_values"].to(device=device, dtype=self.model.dtype)
        pixel_mask = batch["pixel_mask"].to(device=device)

        start = time.perf_counter()
        output_tokens, aux_outputs = self.engine.generate(
            tokens=tokens,
            pos_t=pos_t,
            pos_hw=pos_hw,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            task=task,
        )
        generation_seconds = time.perf_counter() - start

        decoded_output = summarize_decoded_output(
            self.tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=False)
        )
        aux = aux_outputs[0]
        bboxes = pair_bbox_entries(aux.bboxes_raw)
        detections = []
        for index, bbox in enumerate(bboxes):
            detections.append(
                {
                    "index": index,
                    "center": {"x": bbox["x"], "y": bbox["y"]},
                    "height": bbox["h"],
                    "width": bbox["w"],
                    "has_mask": index < len(aux.masks_rle),
                    "engine": "falcon",
                    "label": query,
                    "score": None,
                }
            )

        return {
            "decoded_output": decoded_output,
            "detections": detections,
            "bboxes": bboxes,
            "num_masks": len(aux.masks_rle),
            "masks_rle": aux.masks_rle,
            "generation_seconds": generation_seconds,
        }


class FalconPipelineRealtimeService:
    def __init__(self, options: RuntimeOptions, session: LiveSessionConfig):
        self.options = options
        self.session = session
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.capture: cv2.VideoCapture | None = None
        self.capture_source_key: str | None = None
        self.stream_info: dict[str, Any] | None = None
        self.latest_source_state = build_source_state(
            input_url=session.source_url,
            source_type=infer_source_type(session.source_url),
            status=(SOURCE_STATUS_IDLE if not session.source_url.strip() else SOURCE_STATUS_RESOLVING),
            cookie_file_configured=options.youtube_cookie_file is not None,
        )
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_id = 0
        self.prompt_revision = 0
        self.latest_falcon_guidance: FalconGuidance | None = None
        self.latest_falcon_error: str | None = None
        self.latest_sam3_guidance: Sam3Guidance | None = None
        self.latest_sam3_error: str | None = None
        self.latest_sam3_request: Sam3Request | None = None
        self.latest_result: dict[str, Any] = {}
        self.latest_jpeg: bytes = b""
        self.latest_frame_is_placeholder = True
        self.latest_frame_note = "Waiting for a stream source."
        self.latest_metrics: dict[str, Any] = {
            "capture_state": "idle",
            "pipeline_state": "warming_up",
            "last_error": None,
            "processed_fps": 0.0,
            "last_processed_at": None,
            "latest_generation_seconds": None,
            "falcon_guidance_state": "idle",
            "falcon_guidance_age_seconds": None,
            "falcon_guidance_generation_seconds": None,
            "sam3_segmentation_state": "idle",
            "sam3_segmentation_age_seconds": None,
            "sam3_segmentation_generation_seconds": None,
        }
        self.processed_timestamps: deque[float] = deque(maxlen=40)
        self.falcon_runtime: FalconRealtimeRuntime | None = None
        self.falcon_load_error: str | None = None
        self.rtdetr_runtime: dict[str, Any] | None = None
        self.rtdetr_load_error: str | None = None
        self.sam3_runtime: dict[str, Any] | None = None
        self.sam3_load_error: str | None = None
        self.model_status: dict[str, dict[str, Any]] = {
            "falcon": {
                "requested": True,
                "state": "pending",
                "error": None,
                "error_kind": None,
            },
            "rt_detr": {
                "requested": self.options.load_rtdetr,
                "state": "pending" if self.options.load_rtdetr else "disabled",
                "error": None,
                "error_kind": None,
            },
            "sam3": {
                "requested": self.options.load_sam3,
                "state": "pending" if self.options.load_sam3 else "disabled",
                "error": None,
                "error_kind": None,
            },
        }
        self.threads: list[threading.Thread] = []
        self._set_placeholder_frame("Waiting for a stream source.", clear_result=True)

    def start(self) -> None:
        configure_huggingface_cache(self.options.cache_dir)
        self.options.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_metric("pipeline_state", "loading_models")
        self._set_placeholder_frame("Loading models.", clear_result=True)

        self.threads = [
            threading.Thread(target=self._load_models, name="falcon-loader", daemon=True),
            threading.Thread(target=self._capture_loop, name="falcon-capture", daemon=True),
            threading.Thread(target=self._falcon_guidance_loop, name="falcon-guidance", daemon=True),
            threading.Thread(target=self._sam3_loop, name="falcon-sam3", daemon=True),
            threading.Thread(target=self._process_loop, name="falcon-process", daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None

    def _set_metric(self, key: str, value: Any) -> None:
        with self.lock:
            self.latest_metrics[key] = value

    def _set_model_state(
        self,
        name: str,
        state: str,
        error: str | None = None,
        error_kind: str | None = None,
    ) -> None:
        with self.lock:
            record = self.model_status[name]
            record["state"] = state
            record["error"] = error
            if state == "error":
                record["error_kind"] = error_kind or classify_model_load_error_kind(error)
            else:
                record["error_kind"] = None

    def _set_source_state(self, state: dict[str, Any]) -> None:
        with self.lock:
            self.latest_source_state = copy.deepcopy(state)

    def _set_placeholder_frame(self, message: str, *, clear_result: bool = False) -> None:
        payload = placeholder_frame(message)
        with self.lock:
            self.latest_jpeg = payload
            self.latest_frame_is_placeholder = True
            self.latest_frame_note = message
            if clear_result:
                self.latest_result = {}

    def _mark_live_frame(self) -> None:
        with self.lock:
            self.latest_frame_is_placeholder = False
            self.latest_frame_note = None

    def _models_ready(self, model_status: dict[str, dict[str, Any]]) -> bool:
        for record in model_status.values():
            if record["state"] not in {"loaded", "disabled"}:
                return False
        return True

    def _blocking_engine_errors(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        blocking: list[dict[str, Any]] = []
        for engine in result.get("engines") or []:
            if not engine.get("enabled"):
                continue
            if engine.get("status") != "error":
                continue
            blocking.append(
                {
                    "name": str(engine.get("name") or "unknown"),
                    "reason": (None if engine.get("reason") is None else str(engine.get("reason"))),
                    "error_kind": (
                        engine.get("error_kind")
                        or classify_engine_error_kind(None if engine.get("reason") is None else str(engine.get("reason")))
                        or ERROR_KIND_INFERENCE_RUNTIME
                    ),
                }
            )
        return blocking

    def _derive_operator_error(
        self,
        *,
        result: dict[str, Any],
        source: dict[str, Any],
        metrics: dict[str, Any],
        model_status: dict[str, dict[str, Any]],
    ) -> tuple[str | None, str | None]:
        source_error_kind = source.get("error_kind")
        source_error_message = source.get("error_message")
        if source_error_kind in {ERROR_KIND_SOURCE_AUTH, ERROR_KIND_SOURCE_UNAVAILABLE}:
            return source_error_kind, source_error_message

        for name in ("falcon", "rt_detr", "sam3"):
            record = model_status.get(name) or {}
            if record.get("state") == "error" and record.get("error"):
                return str(record.get("error_kind") or ERROR_KIND_MODEL_LOAD), str(record.get("error"))

        for engine_error in self._blocking_engine_errors(result):
            return str(engine_error.get("error_kind") or ERROR_KIND_INFERENCE_RUNTIME), str(
                engine_error.get("reason") or f"{engine_error['name']} inference failed"
            )

        pipeline_state = metrics.get("pipeline_state")
        last_error = metrics.get("last_error")
        if pipeline_state == "error" and last_error:
            return ERROR_KIND_INFERENCE_RUNTIME, str(last_error)

        return None, None

    def _derive_service_state(
        self,
        *,
        result: dict[str, Any],
        session: dict[str, Any],
        metrics: dict[str, Any],
        model_status: dict[str, dict[str, Any]],
        frame: dict[str, Any],
    ) -> str:
        if model_status["falcon"]["state"] == "error" or metrics.get("pipeline_state") == "error":
            return "error"
        if metrics.get("capture_state") == "error":
            return "error"
        if model_status["sam3"]["state"] == "error":
            return "error"
        if not self._models_ready(model_status):
            return "warming"
        if not session.get("source_url", "").strip():
            return "idle"
        if metrics.get("capture_state") in {"connecting", "reconnecting"}:
            return "connecting"
        degraded = model_status["rt_detr"]["state"] == "error" or not self._sam3_visual_ready(result, metrics)
        degraded = degraded or bool(self._blocking_engine_errors(result))
        if frame["ready"]:
            return "degraded" if degraded else "live"
        return "waiting_for_frame"

    def _sam3_visual_ready(self, result: dict[str, Any], metrics: dict[str, Any]) -> bool:
        if result.get("primary_engine") != "sam3":
            return False
        if metrics.get("sam3_segmentation_state") not in {"ready", "segmenting"}:
            return False
        return bool(result.get("detections") or result.get("masks_rle") or result.get("num_masks"))

    def _load_models(self) -> None:
        self._set_metric("pipeline_state", "loading_models")
        self._set_metric("last_error", None)

        self._set_model_state("falcon", "loading")
        try:
            self.falcon_runtime = FalconRealtimeRuntime(self.options)
        except Exception as exc:
            self.falcon_runtime = None
            self.falcon_load_error = str(exc)
            self._set_model_state("falcon", "error", self.falcon_load_error)
            self._set_metric("pipeline_state", "error")
            self._set_metric("last_error", f"Falcon load failed: {exc}")
            self._set_placeholder_frame(f"Falcon load failed: {exc}", clear_result=True)
            return
        self._set_model_state("falcon", "loaded")

        if self.options.load_rtdetr:
            self._set_model_state("rt_detr", "loading")
            try:
                self.rtdetr_runtime = load_rtdetr_runtime(self.options.rtdetr_model_id)
            except Exception as exc:
                self.rtdetr_runtime = None
                self.rtdetr_load_error = str(exc)
                self._set_model_state("rt_detr", "error", self.rtdetr_load_error)
            else:
                self._set_model_state("rt_detr", "loaded")

        if self.options.load_sam3:
            self._set_model_state("sam3", "loading")
            try:
                self.sam3_runtime = load_sam3_runtime(
                    self.options.sam3_model_id,
                    allow_experimental_non_cuda=False,
                )
            except Exception as exc:
                self.sam3_runtime = None
                self.sam3_load_error = str(exc)
                self._set_model_state("sam3", "error", self.sam3_load_error)
            else:
                self._set_model_state("sam3", "loaded")

        session = self._session_copy()
        if session.source_url.strip():
            with self.lock:
                current_stream_info = copy.deepcopy(self.stream_info)
                capture_state = self.latest_metrics.get("capture_state")
            source_status = SOURCE_STATUS_READY if capture_state == "running" and current_stream_info else SOURCE_STATUS_RESOLVING
            self._set_source_state(
                build_source_state_from_stream_info(
                    input_url=session.source_url,
                    cookie_file=self.options.youtube_cookie_file,
                    stream_info=current_stream_info,
                    status=source_status,
                )
            )
            self._set_metric("pipeline_state", "waiting_for_frame")
            self._set_placeholder_frame("Models loaded. Waiting for first live frame.", clear_result=True)
        else:
            self._set_source_state(
                build_source_state(
                    input_url="",
                    source_type=None,
                    status=SOURCE_STATUS_IDLE,
                    cookie_file_configured=self.options.youtube_cookie_file is not None,
                )
            )
            self._set_metric("pipeline_state", "waiting_for_source")
            self._set_placeholder_frame("Waiting for a stream source.", clear_result=True)

    def _session_copy(self) -> LiveSessionConfig:
        with self.lock:
            return copy.deepcopy(self.session)

    def update_session(self, update: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            source_changed = False
            guidance_changed = False
            for key, value in update.items():
                if value is None or not hasattr(self.session, key):
                    continue
                if key == "task":
                    value = "segmentation"
                elif key == "enable_sam3":
                    value = True
                if getattr(self.session, key) != value:
                    setattr(self.session, key, value)
                    source_changed = source_changed or key == "source_url"
                    guidance_changed = guidance_changed or key in {
                        "source_url",
                        "prompt",
                        "task",
                        "enable_sam3",
                        "falcon_min_dim",
                        "falcon_max_dim",
                        "falcon_max_new_tokens",
                        "falcon_temperature",
                        "sam3_threshold",
                        "sam3_mask_threshold",
                    }
            if guidance_changed:
                self.prompt_revision += 1
                self.latest_falcon_guidance = None
                self.latest_falcon_error = None
                self.latest_sam3_guidance = None
                self.latest_sam3_error = None
                self.latest_sam3_request = None
                self.latest_metrics["falcon_guidance_state"] = "idle"
                self.latest_metrics["falcon_guidance_age_seconds"] = None
                self.session.task = "segmentation"
                self.session.enable_sam3 = True
                self.latest_metrics["sam3_segmentation_state"] = "idle"
                self.latest_metrics["sam3_segmentation_age_seconds"] = None
            if source_changed:
                self.capture_source_key = None
                self.stream_info = None
                self.latest_frame = None
                self.latest_result = {}
                self.latest_metrics["capture_state"] = "reconnecting"
                self.latest_metrics["pipeline_state"] = "waiting_for_source"
                self.latest_metrics["last_processed_at"] = None
                self.latest_metrics["latest_generation_seconds"] = None
                self.latest_metrics["processed_fps"] = 0.0
                self.latest_frame_is_placeholder = True
                self.latest_frame_note = "Switching stream source."
                self.latest_source_state = build_source_state(
                    input_url=self.session.source_url,
                    source_type=infer_source_type(self.session.source_url),
                    status=(SOURCE_STATUS_IDLE if not self.session.source_url.strip() else SOURCE_STATUS_RESOLVING),
                    cookie_file_configured=self.options.youtube_cookie_file is not None,
                )
            if source_changed:
                self._set_placeholder_frame("Switching stream source.", clear_result=True)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            result = copy.deepcopy(self.latest_result)
            metrics = copy.deepcopy(self.latest_metrics)
            session = copy.deepcopy(asdict(self.session))
            source = copy.deepcopy(self.latest_source_state)
            stream = copy.deepcopy(self.stream_info)
            model_status = copy.deepcopy(self.model_status)
            frame = {
                "state": "placeholder" if self.latest_frame_is_placeholder else "live",
                "is_placeholder": self.latest_frame_is_placeholder,
                "ready": not self.latest_frame_is_placeholder,
                "capture_has_frame": self.latest_frame is not None,
                "note": self.latest_frame_note,
                "bytes": len(self.latest_jpeg),
            }
        error_kind, error_message = self._derive_operator_error(
            result=result,
            source=source,
            metrics=metrics,
            model_status=model_status,
        )
        blocking_engine_errors = self._blocking_engine_errors(result)
        sam3_visual_ready = self._sam3_visual_ready(result, metrics)
        integration_ready = self._models_ready(model_status) and frame["ready"] and not blocking_engine_errors and sam3_visual_ready
        readiness = {
            "service_state": self._derive_service_state(
                result=result,
                session=session,
                metrics=metrics,
                model_status=model_status,
                frame=frame,
            ),
            "models_ready": self._models_ready(model_status),
            "capture_connected": metrics.get("capture_state") == "running",
            "integration_ready": integration_ready,
            "full_pipeline_ready": integration_ready,
            "sam3_visual_ready": sam3_visual_ready,
            "blocking_engine_errors": blocking_engine_errors,
            "error_kind": error_kind,
            "error_message": error_message,
        }
        return {
            "session": session,
            "metrics": metrics,
            "source": source,
            "stream": stream,
            "result": result,
            "frame": frame,
            "readiness": readiness,
            "guidelines": PROMPT_GUIDELINES,
            "falcon_available": self.falcon_runtime is not None,
            "falcon_load_error": self.falcon_load_error,
            "rtdetr_available": self.rtdetr_runtime is not None,
            "rtdetr_load_error": self.rtdetr_load_error,
            "sam3_available": self.sam3_runtime is not None,
            "sam3_load_error": self.sam3_load_error,
            "model_status": model_status,
            "models": {
                "falcon": self.options.falcon_model_id,
                "rt_detr": self.options.rtdetr_model_id,
                "sam3": self.options.sam3_model_id,
            },
        }

    def _set_latest_overlay(self, overlay_bgr: np.ndarray, result: dict[str, Any]) -> None:
        with self.lock:
            self.latest_jpeg = encode_jpeg(overlay_bgr, quality=self.session.jpeg_quality)
            self.latest_result = result
            self.latest_metrics["pipeline_state"] = "running"
            self.latest_metrics["latest_generation_seconds"] = result.get("generation_seconds")
            self.latest_metrics["last_processed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._mark_live_frame()

    def get_latest_jpeg(self) -> bytes:
        with self.lock:
            return bytes(self.latest_jpeg)

    def _open_capture(
        self,
        source_url: str,
        timeout_seconds: float,
    ) -> tuple[cv2.VideoCapture | None, dict[str, Any] | None, dict[str, Any], int]:
        source_state, exit_code, stream_info, capture = probe_source(
            source_url,
            cookie_file=self.options.youtube_cookie_file,
            timeout_seconds=timeout_seconds,
            keep_capture_open=True,
        )
        return capture, stream_info, source_state, exit_code

    def _capture_loop(self) -> None:
        read_deadline_started: float | None = None

        while not self.stop_event.is_set():
            session = self._session_copy()
            source_url = session.source_url.strip()

            if not source_url:
                self._set_metric("capture_state", "idle")
                self._set_source_state(
                    build_source_state(
                        input_url="",
                        source_type=None,
                        status=SOURCE_STATUS_IDLE,
                        cookie_file_configured=self.options.youtube_cookie_file is not None,
                    )
                )
                time.sleep(0.25)
                continue

            if self.capture is None or self.capture_source_key != source_url:
                self._set_metric("capture_state", "connecting")
                self._set_source_state(
                    build_source_state(
                        input_url=source_url,
                        source_type=infer_source_type(source_url),
                        status=SOURCE_STATUS_RESOLVING,
                        cookie_file_configured=self.options.youtube_cookie_file is not None,
                    )
                )
                capture, stream_info, source_state, exit_code = self._open_capture(source_url, session.stream_open_timeout)
                if exit_code != 0 or capture is None or stream_info is None:
                    error_message = source_state.get("error_message") or "Could not resolve or open the stream source."
                    self._set_metric("capture_state", "error")
                    self._set_metric("last_error", f"Stream connect failed: {error_message}")
                    self._set_source_state(source_state)
                    self._set_placeholder_frame(error_message, clear_result=True)
                    time.sleep(2.0)
                    continue

                with self.lock:
                    if self.capture is not None:
                        self.capture.release()
                    self.capture = capture
                    self.capture_source_key = source_url
                    self.stream_info = stream_info
                    self.latest_metrics["capture_state"] = "running"
                    self.latest_metrics["last_error"] = None
                self._set_source_state(source_state)
                read_deadline_started = None

            assert self.capture is not None
            ok, frame = self.capture.read()
            if ok and frame is not None and frame.size:
                with self.lock:
                    self.latest_frame = frame
                    self.latest_frame_id += 1
                read_deadline_started = None
                continue

            if read_deadline_started is None:
                read_deadline_started = time.time()
            elif time.time() - read_deadline_started > session.stream_read_timeout:
                with self.lock:
                    if self.capture is not None:
                        self.capture.release()
                    self.capture = None
                    self.capture_source_key = None
                    self.latest_metrics["capture_state"] = "reconnecting"
                    self.latest_metrics["last_error"] = "Stream stalled; reopening video source."
                read_deadline_started = None
            time.sleep(0.05)

    @staticmethod
    def _empty_falcon_inference(message: str) -> dict[str, Any]:
        return {
            "decoded_output": message,
            "detections": [],
            "bboxes": [],
            "num_masks": 0,
            "masks_rle": [],
            "generation_seconds": 0.0,
        }

    def _falcon_guidance_loop(self) -> None:
        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                frame_id = self.latest_frame_id
                prompt_revision = self.prompt_revision
                session = copy.deepcopy(self.session)
                guidance = copy.deepcopy(self.latest_falcon_guidance)

            if frame is None or not session.source_url.strip() or self.falcon_runtime is None:
                time.sleep(0.05)
                continue

            refresh_due = (
                guidance is None
                or guidance.prompt_revision != prompt_revision
                or (time.time() - guidance.ran_at) >= max(session.falcon_refresh_seconds, 0.2)
            )
            if not refresh_due:
                time.sleep(0.05)
                continue

            try:
                self._set_metric("falcon_guidance_state", "refreshing")
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                model_image = resize_image_to_bounds(
                    pil_image,
                    min_dim=session.min_dim,
                    max_dim=session.max_dim,
                )
                assert self.falcon_runtime is not None
                falcon_inference = self.falcon_runtime.run(
                    image=model_image,
                    query=session.prompt,
                    task=falcon_task_for_live_prompt(session.task),
                    min_dim=session.falcon_min_dim,
                    max_dim=session.falcon_max_dim,
                    max_new_tokens=session.falcon_max_new_tokens,
                    temperature=session.falcon_temperature,
                )
                guidance = FalconGuidance(
                    inference=falcon_inference,
                    ran_at=time.time(),
                    frame_id=frame_id,
                    prompt_revision=prompt_revision,
                )
                with self.lock:
                    self.latest_falcon_guidance = guidance
                    self.latest_falcon_error = None
                    self.latest_metrics["falcon_guidance_state"] = "ready"
                    self.latest_metrics["falcon_guidance_age_seconds"] = 0.0
                    self.latest_metrics["falcon_guidance_generation_seconds"] = falcon_inference.get("generation_seconds")
            except Exception as exc:
                with self.lock:
                    self.latest_falcon_error = str(exc)
                    self.latest_metrics["falcon_guidance_state"] = "error"
                    if self.latest_falcon_guidance is None:
                        self.latest_metrics["last_error"] = f"Falcon refresh failed: {exc}"
                time.sleep(0.25)

    def _queue_sam3_request(
        self,
        *,
        image: Image.Image,
        session: LiveSessionConfig,
        prompt_bboxes: list[dict[str, float]],
        frame_id: int,
        prompt_revision: int,
    ) -> None:
        if not prompt_bboxes:
            return
        request = Sam3Request(
            image=image.copy(),
            query=session.prompt,
            prompt_bboxes=copy.deepcopy(prompt_bboxes),
            threshold=session.sam3_threshold,
            mask_threshold=session.sam3_mask_threshold,
            frame_id=frame_id,
            prompt_revision=prompt_revision,
        )
        with self.lock:
            self.latest_sam3_request = request
            if self.latest_metrics.get("sam3_segmentation_state") in {"disabled", "idle"}:
                self.latest_metrics["sam3_segmentation_state"] = "queued"

    def _sam3_loop(self) -> None:
        last_started_key: tuple[int, int] | None = None
        while not self.stop_event.is_set():
            with self.lock:
                request = copy.deepcopy(self.latest_sam3_request)
                session = copy.deepcopy(self.session)

            if self.sam3_runtime is None:
                self._set_metric("sam3_segmentation_state", "unavailable")
                time.sleep(0.10)
                continue

            if request is None:
                self._set_metric("sam3_segmentation_state", "waiting_for_boxes")
                time.sleep(0.05)
                continue

            request_key = (request.prompt_revision, request.frame_id)
            if request_key == last_started_key:
                time.sleep(0.03)
                continue

            last_started_key = request_key
            try:
                self._set_metric("sam3_segmentation_state", "segmenting")
                assert self.sam3_runtime is not None
                sam3_inference = run_sam3_inference(
                    runtime=self.sam3_runtime,
                    image=request.image,
                    query=request.query,
                    prompt_bboxes=request.prompt_bboxes,
                    threshold=request.threshold,
                    mask_threshold=request.mask_threshold,
                )
                guidance = Sam3Guidance(
                    inference=sam3_inference,
                    ran_at=time.time(),
                    frame_id=request.frame_id,
                    prompt_revision=request.prompt_revision,
                )
                with self.lock:
                    self.latest_sam3_guidance = guidance
                    self.latest_sam3_error = None
                    self.latest_metrics["sam3_segmentation_state"] = "ready"
                    self.latest_metrics["sam3_segmentation_age_seconds"] = 0.0
                    self.latest_metrics["sam3_segmentation_generation_seconds"] = sam3_inference.get("generation_seconds")
            except Exception as exc:
                with self.lock:
                    self.latest_sam3_error = str(exc)
                    self.latest_metrics["sam3_segmentation_state"] = "error"
                time.sleep(0.25)

    def _process_loop(self) -> None:
        last_processed_frame_id = -1

        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                frame_id = self.latest_frame_id
                prompt_revision = self.prompt_revision
                session = copy.deepcopy(self.session)
                falcon_guidance = copy.deepcopy(self.latest_falcon_guidance)
                falcon_error = self.latest_falcon_error
                sam3_guidance = copy.deepcopy(self.latest_sam3_guidance)
                sam3_error = self.latest_sam3_error

            if frame is None or not session.source_url.strip():
                time.sleep(0.05)
                continue

            if self.falcon_runtime is None:
                time.sleep(0.05)
                continue

            if frame_id == last_processed_frame_id:
                time.sleep(0.01)
                continue

            try:
                frame_processing_start = time.perf_counter()
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                model_image = resize_image_to_bounds(
                    pil_image,
                    min_dim=session.min_dim,
                    max_dim=session.max_dim,
                )

                if falcon_guidance is not None and falcon_guidance.prompt_revision != prompt_revision:
                    falcon_guidance = None
                if sam3_guidance is not None and sam3_guidance.prompt_revision != prompt_revision:
                    sam3_guidance = None

                falcon_inference = (
                    falcon_guidance.inference
                    if falcon_guidance is not None
                    else self._empty_falcon_inference("Falcon guidance is warming up; using detector-only frame updates.")
                )

                rtdetr_inference = None
                rtdetr_error = self.rtdetr_load_error
                if self.rtdetr_runtime is not None:
                    try:
                        rtdetr_inference = run_rtdetr_inference(
                            runtime=self.rtdetr_runtime,
                            image=model_image,
                            query=session.prompt,
                            falcon_bboxes=falcon_inference.get("bboxes") or [],
                            threshold=session.rtdetr_threshold,
                        )
                        rtdetr_error = None
                    except Exception as exc:
                        rtdetr_error = str(exc)

                sam3_inference = None if sam3_guidance is None else sam3_guidance.inference
                sam3_error = sam3_error or self.sam3_load_error
                prompt_boxes = (
                    (rtdetr_inference or {}).get("bboxes")
                    or falcon_inference.get("bboxes")
                    or []
                )
                if self.sam3_runtime is None:
                    sam3_error = sam3_error or "SAM 3 runtime is unavailable."
                else:
                    self._queue_sam3_request(
                        image=model_image,
                        session=session,
                        prompt_bboxes=prompt_boxes,
                        frame_id=frame_id,
                        prompt_revision=prompt_revision,
                    )
                    if sam3_inference is None:
                        sam3_error = sam3_error or "SAM 3 segmentation is warming up."

                if falcon_guidance is None:
                    effective_falcon_error = falcon_error or "Falcon guidance is warming up."
                elif falcon_error is None:
                    effective_falcon_error = None
                else:
                    effective_falcon_error = f"Latest Falcon refresh failed; using prior guidance. {falcon_error}"

                result = build_orchestrated_inference(
                    query=session.prompt,
                    falcon_inference=falcon_inference,
                    falcon_model_id=self.options.falcon_model_id,
                    falcon_device=self.options.device,
                    falcon_error=effective_falcon_error,
                    rtdetr_inference=rtdetr_inference,
                    rtdetr_model_id=(self.options.rtdetr_model_id if self.options.load_rtdetr else None),
                    rtdetr_error=rtdetr_error,
                    sam3_inference=sam3_inference,
                    sam3_model_id=self.options.sam3_model_id,
                    sam3_error=sam3_error,
                )
                scene_annotations = build_restaurant_scene_annotations(result, session.prompt)
                if scene_annotations is not None:
                    result["scene_annotations"] = scene_annotations
                if falcon_guidance is not None:
                    result["falcon_guidance_age_seconds"] = round(time.time() - falcon_guidance.ran_at, 3)
                    result["falcon_guidance_frame_id"] = falcon_guidance.frame_id
                if sam3_guidance is not None:
                    result["sam3_segmentation_age_seconds"] = round(time.time() - sam3_guidance.ran_at, 3)
                    result["sam3_segmentation_frame_id"] = sam3_guidance.frame_id
                result["frame_generation_seconds"] = time.perf_counter() - frame_processing_start
                result["generation_seconds"] = result["frame_generation_seconds"]

                overlay_bgr = self._render_overlay(frame, result, session)
                self.processed_timestamps.append(time.time())
                fps = 0.0
                if len(self.processed_timestamps) >= 2:
                    elapsed = self.processed_timestamps[-1] - self.processed_timestamps[0]
                    if elapsed > 0:
                        fps = (len(self.processed_timestamps) - 1) / elapsed
                self._set_metric("processed_fps", round(fps, 2))
                self._set_metric("last_error", effective_falcon_error if falcon_guidance is None else None)
                if falcon_guidance is not None:
                    self._set_metric("falcon_guidance_age_seconds", round(time.time() - falcon_guidance.ran_at, 3))
                if sam3_guidance is not None:
                    self._set_metric("sam3_segmentation_age_seconds", round(time.time() - sam3_guidance.ran_at, 3))
                self._set_latest_overlay(overlay_bgr, result)
                last_processed_frame_id = frame_id
            except Exception as exc:
                self._set_metric("pipeline_state", "error")
                self._set_metric("last_error", f"Processing failed: {exc}")
                self._set_placeholder_frame(f"Processing failed: {exc}")
                time.sleep(0.25)

    def _render_overlay(
        self,
        frame_bgr: np.ndarray,
        inference: dict[str, Any],
        session: LiveSessionConfig,
    ) -> np.ndarray:
        scene_annotations = inference.get("scene_annotations") or {}
        entities = scene_annotations.get("entities") or []
        overlay_pil = render_visualization(
            Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
            [] if entities else (inference.get("bboxes") or []),
            inference.get("masks_rle") or [],
            interior_opacity=0.22,
            border_thickness=2,
        )
        overlay_bgr = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGB2BGR)
        height, width = overlay_bgr.shape[:2]

        header = overlay_bgr.copy()
        cv2.rectangle(header, (0, 0), (width, 124), (20, 47, 48), -1)
        overlay_bgr = cv2.addWeighted(header, 0.42, overlay_bgr, 0.58, 0.0)

        detections = inference.get("detections") or []
        primary_engine = str(inference.get("primary_engine") or "falcon").replace("_", "-").upper()
        latency = inference.get("generation_seconds")
        latency_label = "n/a" if latency is None else f"{float(latency):.2f}s"
        stream_title = ""
        if self.stream_info:
            stream_title = self.stream_info.get("title") or self.stream_info.get("requested_url") or ""

        status_lines = [
            f"FALCON PIPELINE // {primary_engine}",
            f"Prompt: {session.prompt}",
            f"Task: {session.task}   Detections: {len(detections)}   End-to-end: {latency_label}",
        ]
        if scene_annotations:
            counts = scene_annotations.get("counts") or {}
            status_lines.append(
                "Restaurant mode: "
                f"goers {counts.get('restaurant_goers', 0)}   "
                f"servers {counts.get('servers', 0)}   "
                f"unclassified {counts.get('unclassified', 0)}   "
                f"tables {counts.get('tables', 0)}   "
                f"needs service {counts.get('needs_service', 0)}"
            )
        if stream_title:
            status_lines.append(f"Source: {stream_title[:100]}")

        y = 38
        for index, line in enumerate(status_lines):
            font_scale = 0.95 if index == 0 else 0.68
            thickness = 2 if index == 0 else 1
            cv2.putText(
                overlay_bgr,
                line,
                (28, y),
                cv2.FONT_HERSHEY_DUPLEX if index == 0 else cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (248, 237, 209),
                thickness,
                cv2.LINE_AA,
            )
            y += 32

        for entity in entities[:48]:
            bbox = entity.get("bbox") or {}
            x0, y0, x1, y1 = bbox_norm_xyxy(bbox, width, height)
            color = entity_color_bgr(entity)
            role = str(entity.get("role") or entity.get("kind") or "entity").replace("_", " ")
            if entity.get("needs_service"):
                role = f"{role} needs service"
            cv2.rectangle(overlay_bgr, (x0, y0), (x1, y1), color, 2)
            text = role.upper()
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            text_width, text_height = text_size
            label_y = max(y0 - 10, 24)
            cv2.rectangle(
                overlay_bgr,
                (x0, label_y - text_height - 10),
                (x0 + text_width + 12, label_y + 4),
                color,
                -1,
            )
            cv2.putText(
                overlay_bgr,
                text,
                (x0 + 6, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 249, 237),
                2,
                cv2.LINE_AA,
            )

        for detection in ([] if entities else detections[:24]):
            x0, y0, x1, y1 = detection_bbox_xyxy(detection, width, height)
            label = detection.get("label") or session.prompt
            score = detection.get("score")
            if score is None:
                text = str(label)
            else:
                text = f"{label} {float(score):.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            text_width, text_height = text_size
            label_y = max(y0 - 10, 24)
            cv2.rectangle(
                overlay_bgr,
                (x0, label_y - text_height - 10),
                (x0 + text_width + 12, label_y + 4),
                (13, 124, 119),
                -1,
            )
            cv2.putText(
                overlay_bgr,
                text,
                (x0 + 6, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (255, 249, 237),
                2,
                cv2.LINE_AA,
            )

        return overlay_bgr

    def mjpeg_stream(self):
        boundary = b"--frame"
        while not self.stop_event.is_set():
            with self.lock:
                payload = self.latest_jpeg
            yield (
                boundary
                + b"\r\n"
                + b"Content-Type: image/jpeg\r\n"
                + f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8")
                + payload
                + b"\r\n"
            )
            time.sleep(0.04)


def load_ui_html(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<html><body><h1>Falcon Pipeline Realtime UI missing.</h1></body></html>"


def run_source_preflight(
    *,
    source_url: str,
    cookie_file: Path | None,
    timeout_seconds: float,
) -> tuple[dict[str, Any], int]:
    source_state, exit_code, _stream_info, _capture = probe_source(
        source_url,
        cookie_file=cookie_file,
        timeout_seconds=timeout_seconds,
        keep_capture_open=False,
    )
    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "source": source_state,
    }, exit_code


def run_sam_preflight(*, model_id: str) -> tuple[dict[str, Any], int]:
    torch_stack = inspect_torch_stack()
    token_configured = bool(huggingface_token())
    base_payload: dict[str, Any] = {
        "ok": False,
        "exit_code": 25,
        "model_id": model_id,
        "status": SAM_PREFLIGHT_STATUS_LOAD_FAILED,
        "error_kind": ERROR_KIND_MODEL_LOAD,
        "error_message": None,
        "token_configured": token_configured,
        "cuda_available": bool(torch_stack.get("cuda_available")),
        "transformers_sam3_available": bool(
            torch_stack.get("transformers_available") and torch_stack.get("has_sam3")
        ),
    }

    if not token_configured:
        payload = {
            **base_payload,
            "exit_code": 22,
            "status": SAM_PREFLIGHT_STATUS_MISSING_TOKEN,
            "error_kind": ERROR_KIND_MODEL_ACCESS,
            "error_message": "SAM 3 is mandatory; set HF_TOKEN or HUGGING_FACE_HUB_TOKEN.",
        }
        return payload, 22

    if not base_payload["transformers_sam3_available"]:
        payload = {
            **base_payload,
            "exit_code": 24,
            "status": SAM_PREFLIGHT_STATUS_UNSUPPORTED,
            "error_kind": ERROR_KIND_MODEL_UNSUPPORTED,
            "error_message": "This image does not expose Transformers Sam3Model/Sam3Processor support.",
        }
        return payload, 24

    try:
        runtime = load_sam3_runtime(model_id, allow_experimental_non_cuda=False)
    except ModelAccessError as exc:
        payload = {
            **base_payload,
            "exit_code": 23,
            "status": SAM_PREFLIGHT_STATUS_ACCESS_REQUIRED,
            "error_kind": ERROR_KIND_MODEL_ACCESS,
            "error_message": str(exc),
        }
        return payload, 23
    except ModelUnsupportedError as exc:
        payload = {
            **base_payload,
            "exit_code": 24,
            "status": SAM_PREFLIGHT_STATUS_UNSUPPORTED,
            "error_kind": ERROR_KIND_MODEL_UNSUPPORTED,
            "error_message": str(exc),
        }
        return payload, 24
    except Exception as exc:
        error_kind = classify_model_load_error_kind(str(exc))
        if error_kind == ERROR_KIND_MODEL_ACCESS:
            exit_code = 23
            status = SAM_PREFLIGHT_STATUS_ACCESS_REQUIRED
        elif error_kind == ERROR_KIND_MODEL_UNSUPPORTED:
            exit_code = 24
            status = SAM_PREFLIGHT_STATUS_UNSUPPORTED
        else:
            exit_code = 25
            status = SAM_PREFLIGHT_STATUS_LOAD_FAILED
            error_kind = ERROR_KIND_MODEL_LOAD
        payload = {
            **base_payload,
            "exit_code": exit_code,
            "status": status,
            "error_kind": error_kind,
            "error_message": str(exc),
        }
        return payload, exit_code

    payload = {
        **base_payload,
        "ok": True,
        "exit_code": 0,
        "status": SAM_PREFLIGHT_STATUS_READY,
        "error_kind": None,
        "error_message": None,
        "cuda_available": True,
        "transformers_sam3_available": True,
    }
    if isinstance(runtime, dict) and runtime.get("device"):
        payload["device"] = runtime.get("device")
    return payload, 0


def create_app(service: FalconPipelineRealtimeService, ui_html: str):
    app = FastAPI(title="Falcon Pipeline Realtime")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(ui_html)

    @app.get("/api/state")
    async def state() -> JSONResponse:
        return JSONResponse(service.get_state())

    @app.post("/api/session")
    async def update_session(request: Request) -> JSONResponse:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
        return JSONResponse(service.update_session(payload))

    @app.get("/api/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "falcon-pipeline-realtime",
            }
        )

    @app.get("/api/stream.mjpg")
    async def stream() -> StreamingResponse:
        return StreamingResponse(
            service.mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/frame.jpg")
    async def frame() -> Response:
        return Response(content=service.get_latest_jpeg(), media_type="image/jpeg")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the Falcon Pipeline realtime web app for CUDA-backed livestream perception. "
            "This version is designed for an L4-class Google Cloud GPU VM."
        )
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port.")
    parser.add_argument("--source-url", type=str, default="", help="Optional initial livestream URL.")
    parser.add_argument("--prompt", type=str, default="people under the umbrellas", help="Initial Falcon prompt.")
    parser.add_argument(
        "--task",
        choices=("segmentation",),
        default="segmentation",
        help="Primary live task mode. SAM 3 segmentation is mandatory for this service.",
    )
    parser.add_argument("--enable-sam3", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--falcon-model-id", default=PERCEPTION_300M_MODEL_ID, help="Falcon Perception model id.")
    parser.add_argument("--rtdetr-model-id", default=DEFAULT_RT_DETR_MODEL_ID, help="RT-DETR model id.")
    parser.add_argument("--sam3-model-id", default=DEFAULT_SAM3_MODEL_ID, help="SAM 3 model id.")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Falcon torch dtype. L4 supports bfloat16.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for Falcon orchestration.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="HF cache directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact output directory.")
    parser.add_argument(
        "--yt-cookies-file",
        type=Path,
        default=None,
        help="Optional Netscape-format cookies.txt file for YouTube streams that require authentication.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Resolve and probe the source, emit JSON to stdout, then exit without loading models.",
    )
    parser.add_argument(
        "--sam-preflight-only",
        action="store_true",
        help="Probe the configured SAM 3 checkpoint, emit JSON to stdout, then exit without launching the service.",
    )
    parser.add_argument("--ui-path", type=Path, default=DEFAULT_UI_PATH, help="HTML UI file to serve.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for Falcon.")
    parser.add_argument("--no-load-rtdetr", action="store_true", help="Skip loading RT-DETR at startup.")
    parser.add_argument("--load-sam3", dest="load_sam3", action="store_true", default=True, help=argparse.SUPPRESS)
    parser.add_argument("--min-dim", type=int, default=480, help="Shortest side used for live detector frames.")
    parser.add_argument("--max-dim", type=int, default=960, help="Longest side used for live detector frames.")
    parser.add_argument("--falcon-min-dim", type=int, default=256, help="Shortest side for Falcon refresh passes.")
    parser.add_argument("--falcon-max-dim", type=int, default=640, help="Longest side for Falcon refresh passes.")
    parser.add_argument("--falcon-max-new-tokens", type=int, default=128, help="Falcon max new tokens per refresh.")
    parser.add_argument("--falcon-refresh-seconds", type=float, default=5.0, help="Seconds between Falcon refreshes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    resolved_cookie_file = (None if args.yt_cookies_file is None else args.yt_cookies_file.expanduser().resolve())
    if args.sam_preflight_only:
        payload, exit_code = run_sam_preflight(model_id=args.sam3_model_id)
        print(json.dumps(payload, indent=2))
        return exit_code

    if args.preflight_only:
        payload, exit_code = run_source_preflight(
            source_url=args.source_url,
            cookie_file=resolved_cookie_file,
            timeout_seconds=LiveSessionConfig().stream_open_timeout,
        )
        print(json.dumps(payload, indent=2))
        return exit_code

    import uvicorn

    options = RuntimeOptions(
        cache_dir=args.cache_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        youtube_cookie_file=resolved_cookie_file,
        falcon_model_id=args.falcon_model_id,
        rtdetr_model_id=args.rtdetr_model_id,
        sam3_model_id=args.sam3_model_id,
        dtype=args.dtype,
        device=args.device,
        compile_model=not args.no_compile,
        load_rtdetr=not args.no_load_rtdetr,
        load_sam3=True,
        host=args.host,
        port=args.port,
        ui_path=args.ui_path.expanduser().resolve(),
    )
    session = LiveSessionConfig(
        source_url=args.source_url,
        prompt=args.prompt,
        task="segmentation",
        enable_sam3=True,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        falcon_min_dim=args.falcon_min_dim,
        falcon_max_dim=args.falcon_max_dim,
        falcon_max_new_tokens=args.falcon_max_new_tokens,
        falcon_refresh_seconds=args.falcon_refresh_seconds,
    )

    service = FalconPipelineRealtimeService(options, session)
    service.start()
    app = create_app(service, load_ui_html(options.ui_path))

    try:
        uvicorn.run(app, host=options.host, port=options.port, log_level="info")
    finally:
        service.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
