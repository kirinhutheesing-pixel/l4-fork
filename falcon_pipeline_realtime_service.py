#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
    build_orchestrated_inference,
    load_rtdetr_runtime,
    load_sam3_runtime,
    run_rtdetr_inference,
    run_sam3_inference,
)
from run_falcon_pipeline import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    configure_huggingface_cache,
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
    task: str = "detection"
    enable_sam3: bool = False
    min_dim: int = 480
    max_dim: int = 960
    falcon_min_dim: int = 256
    falcon_max_dim: int = 640
    falcon_max_new_tokens: int = 128
    falcon_temperature: float = 0.0
    falcon_refresh_seconds: float = 2.0
    rtdetr_threshold: float = 0.35
    sam3_threshold: float = 0.5
    sam3_mask_threshold: float = 0.5
    stream_open_timeout: float = 60.0
    stream_read_timeout: float = 20.0
    jpeg_quality: int = 85


@dataclass
class RuntimeOptions:
    cache_dir: Path
    output_dir: Path
    falcon_model_id: str = PERCEPTION_300M_MODEL_ID
    rtdetr_model_id: str = DEFAULT_RT_DETR_MODEL_ID
    sam3_model_id: str = DEFAULT_SAM3_MODEL_ID
    dtype: str = "bfloat16"
    device: str = "cuda"
    compile_model: bool = True
    load_rtdetr: bool = True
    load_sam3: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    ui_path: Path = DEFAULT_UI_PATH


@dataclass
class FalconGuidance:
    inference: dict[str, Any]
    ran_at: float
    frame_id: int


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
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_id = 0
        self.prompt_revision = 0
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
            },
            "rt_detr": {
                "requested": self.options.load_rtdetr,
                "state": "pending" if self.options.load_rtdetr else "disabled",
                "error": None,
            },
            "sam3": {
                "requested": self.options.load_sam3,
                "state": "pending" if self.options.load_sam3 else "disabled",
                "error": None,
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

    def _set_model_state(self, name: str, state: str, error: str | None = None) -> None:
        with self.lock:
            record = self.model_status[name]
            record["state"] = state
            record["error"] = error

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

    def _derive_service_state(
        self,
        *,
        session: dict[str, Any],
        metrics: dict[str, Any],
        model_status: dict[str, dict[str, Any]],
        frame: dict[str, Any],
    ) -> str:
        if model_status["falcon"]["state"] == "error" or metrics.get("pipeline_state") == "error":
            return "error"
        if metrics.get("capture_state") == "error":
            return "error"
        if not self._models_ready(model_status):
            return "warming"
        if not session.get("source_url", "").strip():
            return "idle"
        if metrics.get("capture_state") in {"connecting", "reconnecting"}:
            return "connecting"
        degraded = model_status["rt_detr"]["state"] == "error" or (
            bool(session.get("enable_sam3")) and model_status["sam3"]["state"] == "error"
        )
        if frame["ready"]:
            return "degraded" if degraded else "live"
        return "waiting_for_frame"

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
            self._set_metric("pipeline_state", "waiting_for_frame")
            self._set_placeholder_frame("Models loaded. Waiting for first live frame.", clear_result=True)
        else:
            self._set_metric("pipeline_state", "waiting_for_source")
            self._set_placeholder_frame("Waiting for a stream source.", clear_result=True)

    def _session_copy(self) -> LiveSessionConfig:
        with self.lock:
            return copy.deepcopy(self.session)

    def update_session(self, update: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            source_changed = False
            prompt_changed = False
            for key, value in update.items():
                if value is None or not hasattr(self.session, key):
                    continue
                if key == "task" and value not in {"detection", "segmentation"}:
                    continue
                if getattr(self.session, key) != value:
                    setattr(self.session, key, value)
                    source_changed = source_changed or key == "source_url"
                    prompt_changed = prompt_changed or key == "prompt"
            if prompt_changed:
                self.prompt_revision += 1
            if source_changed:
                self.prompt_revision += 1
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
            if source_changed:
                self._set_placeholder_frame("Switching stream source.", clear_result=True)
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        with self.lock:
            result = copy.deepcopy(self.latest_result)
            metrics = copy.deepcopy(self.latest_metrics)
            session = copy.deepcopy(asdict(self.session))
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
        readiness = {
            "service_state": self._derive_service_state(
                session=session,
                metrics=metrics,
                model_status=model_status,
                frame=frame,
            ),
            "models_ready": self._models_ready(model_status),
            "capture_connected": metrics.get("capture_state") == "running",
            "integration_ready": self._models_ready(model_status) and frame["ready"],
        }
        return {
            "session": session,
            "metrics": metrics,
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

    def _open_capture(self, source_url: str, timeout_seconds: float) -> tuple[cv2.VideoCapture, dict[str, Any]]:
        stream_info = resolve_stream_source(source_url)
        capture = open_video_capture(stream_info, timeout_seconds)
        return capture, stream_info

    def _capture_loop(self) -> None:
        read_deadline_started: float | None = None

        while not self.stop_event.is_set():
            session = self._session_copy()
            source_url = session.source_url.strip()

            if not source_url:
                self._set_metric("capture_state", "idle")
                time.sleep(0.25)
                continue

            if self.capture is None or self.capture_source_key != source_url:
                self._set_metric("capture_state", "connecting")
                try:
                    capture, stream_info = self._open_capture(source_url, session.stream_open_timeout)
                except Exception as exc:
                    self._set_metric("capture_state", "error")
                    self._set_metric("last_error", f"Stream connect failed: {exc}")
                    self._set_placeholder_frame(str(exc), clear_result=True)
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

    def _process_loop(self) -> None:
        last_processed_frame_id = -1
        last_prompt_revision = -1
        falcon_guidance: FalconGuidance | None = None

        while not self.stop_event.is_set():
            with self.lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                frame_id = self.latest_frame_id
                prompt_revision = self.prompt_revision
                session = copy.deepcopy(self.session)

            if frame is None or not session.source_url.strip():
                time.sleep(0.05)
                continue

            if self.falcon_runtime is None:
                time.sleep(0.05)
                continue

            if frame_id == last_processed_frame_id:
                time.sleep(0.01)
                continue

            if prompt_revision != last_prompt_revision:
                falcon_guidance = None
                last_prompt_revision = prompt_revision

            try:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                model_image = resize_image_to_bounds(
                    pil_image,
                    min_dim=session.min_dim,
                    max_dim=session.max_dim,
                )

                falcon_refresh_due = (
                    falcon_guidance is None
                    or (time.time() - falcon_guidance.ran_at) >= max(session.falcon_refresh_seconds, 0.2)
                )
                falcon_error: str | None = None
                if falcon_refresh_due:
                    try:
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
                        falcon_guidance = FalconGuidance(
                            inference=falcon_inference,
                            ran_at=time.time(),
                            frame_id=frame_id,
                        )
                    except Exception as exc:
                        falcon_error = str(exc)

                falcon_inference = (
                    falcon_guidance.inference
                    if falcon_guidance is not None
                    else {
                        "decoded_output": "Falcon guidance unavailable for this frame.",
                        "detections": [],
                        "bboxes": [],
                        "num_masks": 0,
                        "masks_rle": [],
                        "generation_seconds": 0.0,
                    }
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

                sam3_inference = None
                sam3_error = self.sam3_load_error
                if session.enable_sam3 and session.task == "segmentation":
                    prompt_boxes = (
                        rtdetr_inference.get("bboxes")
                        or falcon_inference.get("bboxes")
                        or []
                    )
                    if self.sam3_runtime is None:
                        sam3_error = sam3_error or "SAM 3 runtime is unavailable."
                    else:
                        sam3_inference = run_sam3_inference(
                            runtime=self.sam3_runtime,
                            image=model_image,
                            query=session.prompt,
                            prompt_bboxes=prompt_boxes,
                            threshold=session.sam3_threshold,
                            mask_threshold=session.sam3_mask_threshold,
                        )

                result = build_orchestrated_inference(
                    query=session.prompt,
                    falcon_inference=falcon_inference,
                    falcon_model_id=self.options.falcon_model_id,
                    falcon_device=self.options.device,
                    falcon_error=(
                        falcon_error
                        if falcon_guidance is None
                        else (
                            None
                            if falcon_error is None
                            else f"Latest Falcon refresh failed; using prior guidance. {falcon_error}"
                        )
                    ),
                    rtdetr_inference=rtdetr_inference,
                    rtdetr_model_id=(self.options.rtdetr_model_id if self.options.load_rtdetr else None),
                    rtdetr_error=rtdetr_error,
                    sam3_inference=sam3_inference,
                    sam3_model_id=(self.options.sam3_model_id if session.enable_sam3 else None),
                    sam3_error=sam3_error,
                )
                if falcon_guidance is not None:
                    result["falcon_guidance_age_seconds"] = round(time.time() - falcon_guidance.ran_at, 3)

                overlay_bgr = self._render_overlay(frame, result, session)
                self.processed_timestamps.append(time.time())
                fps = 0.0
                if len(self.processed_timestamps) >= 2:
                    elapsed = self.processed_timestamps[-1] - self.processed_timestamps[0]
                    if elapsed > 0:
                        fps = (len(self.processed_timestamps) - 1) / elapsed
                self._set_metric("processed_fps", round(fps, 2))
                self._set_metric("last_error", falcon_error if falcon_guidance is None else None)
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
        overlay_pil = render_visualization(
            Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)),
            inference.get("bboxes") or [],
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

        for detection in detections[:24]:
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


def create_app(service: FalconPipelineRealtimeService, ui_html: str):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

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
        choices=("detection", "segmentation"),
        default="detection",
        help="Primary live task mode.",
    )
    parser.add_argument("--enable-sam3", action="store_true", help="Enable SAM 3 by default in the UI session.")
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
    parser.add_argument("--ui-path", type=Path, default=DEFAULT_UI_PATH, help="HTML UI file to serve.")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile for Falcon.")
    parser.add_argument("--no-load-rtdetr", action="store_true", help="Skip loading RT-DETR at startup.")
    sam3_group = parser.add_mutually_exclusive_group()
    sam3_group.add_argument("--load-sam3", dest="load_sam3", action="store_true", help="Preload SAM 3 at startup.")
    sam3_group.add_argument("--no-load-sam3", dest="load_sam3", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(load_sam3=False)
    parser.add_argument("--min-dim", type=int, default=480, help="Shortest side used for live detector frames.")
    parser.add_argument("--max-dim", type=int, default=960, help="Longest side used for live detector frames.")
    parser.add_argument("--falcon-min-dim", type=int, default=256, help="Shortest side for Falcon refresh passes.")
    parser.add_argument("--falcon-max-dim", type=int, default=640, help="Longest side for Falcon refresh passes.")
    parser.add_argument("--falcon-max-new-tokens", type=int, default=128, help="Falcon max new tokens per refresh.")
    parser.add_argument("--falcon-refresh-seconds", type=float, default=2.0, help="Seconds between Falcon refreshes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import uvicorn

    options = RuntimeOptions(
        cache_dir=args.cache_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        falcon_model_id=args.falcon_model_id,
        rtdetr_model_id=args.rtdetr_model_id,
        sam3_model_id=args.sam3_model_id,
        dtype=args.dtype,
        device=args.device,
        compile_model=not args.no_compile,
        load_rtdetr=not args.no_load_rtdetr,
        load_sam3=args.load_sam3,
        host=args.host,
        port=args.port,
        ui_path=args.ui_path.expanduser().resolve(),
    )
    session = LiveSessionConfig(
        source_url=args.source_url,
        prompt=args.prompt,
        task=args.task,
        enable_sam3=args.enable_sam3,
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
