from __future__ import annotations

import re
import time
from copy import deepcopy
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_RT_DETR_MODEL_ID = "PekingU/rtdetr_r18vd"
DEFAULT_SAM3_MODEL_ID = "facebook/sam3"


def normalize_engine_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def make_engine_record(
    *,
    name: str,
    enabled: bool,
    status: str,
    available: bool,
    device: str | None = None,
    model_id: str | None = None,
    reason: str | None = None,
    experimental: bool = False,
    detections_count: int = 0,
    num_masks: int = 0,
    generation_seconds: float | None = None,
    prompt_boxes_count: int | None = None,
) -> dict[str, Any]:
    return {
        "name": normalize_engine_name(name),
        "enabled": enabled,
        "status": status,
        "available": available,
        "device": device,
        "model_id": model_id,
        "reason": reason,
        "experimental": experimental,
        "detections_count": detections_count,
        "num_masks": num_masks,
        "generation_seconds": generation_seconds,
        "prompt_boxes_count": prompt_boxes_count,
    }


def _optional_module(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:  # pragma: no cover - surfaced in the UI instead.
        return None, str(exc)


def inspect_torch_stack() -> dict[str, Any]:
    torch, torch_error = _optional_module("torch")
    transformers, transformers_error = _optional_module("transformers")

    if torch is None:
        return {
            "torch_available": False,
            "transformers_available": transformers is not None,
            "device": None,
            "device_label": "Unavailable",
            "torch_version": None,
            "transformers_version": getattr(transformers, "__version__", None),
            "has_rtdetr": False,
            "has_sam3": False,
            "cuda_available": False,
            "mps_available": False,
            "reason": torch_error or transformers_error or "PyTorch is not installed.",
        }

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    if cuda_available:
        device = "cuda"
    elif mps_available:
        device = "mps"
    else:
        device = "cpu"

    has_rtdetr = bool(transformers and hasattr(transformers, "RTDetrForObjectDetection"))
    has_sam3 = bool(transformers and hasattr(transformers, "Sam3Model"))
    return {
        "torch_available": True,
        "transformers_available": transformers is not None,
        "device": device,
        "device_label": device.upper(),
        "torch_version": getattr(torch, "__version__", None),
        "transformers_version": getattr(transformers, "__version__", None),
        "has_rtdetr": has_rtdetr,
        "has_sam3": has_sam3,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "reason": transformers_error,
    }


def inspect_engine_capabilities(*, allow_experimental_sam3: bool = False) -> dict[str, dict[str, Any]]:
    torch_stack = inspect_torch_stack()
    base_reason = torch_stack["reason"]

    rtdetr_available = (
        torch_stack["torch_available"]
        and torch_stack["transformers_available"]
        and torch_stack["has_rtdetr"]
    )
    rtdetr_reason = None if rtdetr_available else (
        base_reason or "Install torch and transformers with RT-DETR support."
    )

    sam3_backend_ready = torch_stack["torch_available"] and torch_stack["transformers_available"] and torch_stack["has_sam3"]
    sam3_available = sam3_backend_ready and (torch_stack["cuda_available"] or allow_experimental_sam3)
    if sam3_available:
        sam3_reason = None
    elif not sam3_backend_ready:
        sam3_reason = base_reason or "Install torch and a recent transformers build with SAM 3 support."
    elif torch_stack["cuda_available"]:
        sam3_reason = None
    elif allow_experimental_sam3:
        sam3_reason = None
    else:
        sam3_reason = (
            "Disabled by default on this device. The official SAM 3 repo targets CUDA GPUs, "
            "and this Mac would need an experimental non-CUDA run."
        )

    return {
        "falcon": make_engine_record(
            name="falcon",
            enabled=True,
            status="ready",
            available=True,
            device="mlx",
            reason=None,
            model_id=None,
        ),
        "rt_detr": make_engine_record(
            name="rt_detr",
            enabled=False,
            status="ready" if rtdetr_available else "unavailable",
            available=rtdetr_available,
            device=torch_stack["device"],
            reason=rtdetr_reason,
            model_id=DEFAULT_RT_DETR_MODEL_ID,
        ),
        "sam3": make_engine_record(
            name="sam3",
            enabled=False,
            status="ready" if sam3_available else "unavailable",
            available=sam3_available,
            device="cuda" if torch_stack["cuda_available"] else torch_stack["device"],
            reason=sam3_reason,
            model_id=DEFAULT_SAM3_MODEL_ID,
            experimental=allow_experimental_sam3 and not torch_stack["cuda_available"],
        ),
    }


def _torch_device():
    torch, _ = _optional_module("torch")
    if torch is None:
        raise RuntimeError("PyTorch is not installed in this environment.")
    if torch.cuda.is_available():
        return torch, torch.device("cuda")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch, torch.device("mps")
    return torch, torch.device("cpu")


def load_rtdetr_runtime(model_id: str = DEFAULT_RT_DETR_MODEL_ID) -> dict[str, Any]:
    torch, device = _torch_device()
    try:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
    except Exception as exc:  # pragma: no cover - dependency surfaced in UI.
        raise RuntimeError("RT-DETR support requires a recent transformers install.") from exc

    processor = RTDetrImageProcessor.from_pretrained(model_id)
    model = RTDetrForObjectDetection.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return {
        "model_id": model_id,
        "device": str(device),
        "torch": torch,
        "model": model,
        "processor": processor,
    }


def _clean_words(value: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", value.lower())
    cleaned: set[str] = set()
    for word in words:
        cleaned.add(word)
        if word.endswith("es") and len(word) > 3:
            cleaned.add(word[:-2])
        if word.endswith("s") and len(word) > 3:
            cleaned.add(word[:-1])
    return cleaned


def query_matches_label(query: str, label: str) -> bool:
    query_words = _clean_words(query)
    label_words = _clean_words(label)
    return bool(query_words & label_words) or label.lower() in query.lower()


def bbox_xyxy_to_center_wh_norm(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> dict[str, float]:
    box_width = max(x1 - x0, 0.0)
    box_height = max(y1 - y0, 0.0)
    center_x = x0 + box_width / 2.0
    center_y = y0 + box_height / 2.0
    return {
        "x": center_x / max(width, 1),
        "y": center_y / max(height, 1),
        "w": box_width / max(width, 1),
        "h": box_height / max(height, 1),
    }


def bbox_center_wh_norm_to_xyxy(
    bbox: dict[str, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    center_x = float(bbox["x"]) * width
    center_y = float(bbox["y"]) * height
    box_width = float(bbox["w"]) * width
    box_height = float(bbox["h"]) * height
    return (
        center_x - box_width / 2.0,
        center_y - box_height / 2.0,
        center_x + box_width / 2.0,
        center_y + box_height / 2.0,
    )


def intersection_over_union(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
    width: int = 1,
    height: int = 1,
) -> float:
    ax0, ay0, ax1, ay1 = bbox_center_wh_norm_to_xyxy(bbox_a, width, height)
    bx0, by0, bx1, by1 = bbox_center_wh_norm_to_xyxy(bbox_b, width, height)

    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    intersection = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _serialize_detection(
    *,
    index: int,
    bbox: dict[str, float],
    label: str | None,
    score: float | None,
    has_mask: bool,
    engine: str,
    support_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "center": {"x": bbox["x"], "y": bbox["y"]},
        "height": bbox["h"],
        "width": bbox["w"],
        "has_mask": has_mask,
        "engine": normalize_engine_name(engine),
        "label": label,
        "score": score,
        "support_reason": support_reason,
    }


def run_rtdetr_inference(
    *,
    runtime: dict[str, Any],
    image: Image.Image,
    query: str,
    falcon_bboxes: list[dict[str, float]],
    threshold: float = 0.35,
    overlap_threshold: float = 0.3,
) -> dict[str, Any]:
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]
    width, height = image.size

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    target_sizes = torch.tensor([(height, width)], device=device)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    generation_seconds = time.perf_counter() - start

    post_processed = processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )[0]

    all_detections: list[dict[str, Any]] = []
    filtered_detections: list[dict[str, Any]] = []
    filtered_bboxes: list[dict[str, float]] = []
    max_overlap = 0.0
    for index, (score_tensor, label_tensor, box_tensor) in enumerate(
        zip(post_processed["scores"], post_processed["labels"], post_processed["boxes"])
    ):
        label_id = int(label_tensor.item())
        label = str(model.config.id2label[label_id])
        score = float(score_tensor.item())
        x0, y0, x1, y1 = [float(value) for value in box_tensor.tolist()]
        bbox = bbox_xyxy_to_center_wh_norm(x0, y0, x1, y1, width, height)

        label_match = query_matches_label(query, label)
        overlap = max((intersection_over_union(bbox, candidate) for candidate in falcon_bboxes), default=0.0)
        max_overlap = max(max_overlap, overlap)

        detection = _serialize_detection(
            index=index,
            bbox=bbox,
            label=label,
            score=score,
            has_mask=False,
            engine="rt_detr",
            support_reason="query label" if label_match else ("falcon overlap" if overlap >= overlap_threshold else None),
        )
        all_detections.append(detection)

        if label_match or (falcon_bboxes and overlap >= overlap_threshold):
            filtered_detections.append(detection)
            filtered_bboxes.append(bbox)

    summary = (
        f"RT-DETR found {len(all_detections)} COCO objects and kept {len(filtered_detections)} "
        f"as Falcon-selected backbone detections for '{query}'."
    )
    return {
        "engine": "rt_detr",
        "model_id": runtime["model_id"],
        "device": runtime["device"],
        "decoded_output": summary,
        "detections": filtered_detections,
        "candidate_detections": all_detections,
        "bboxes": filtered_bboxes,
        "num_masks": 0,
        "masks_rle": [],
        "generation_seconds": generation_seconds,
        "max_overlap": max_overlap,
    }


def _encode_binary_mask(mask_array: np.ndarray) -> dict[str, Any]:
    try:
        from pycocotools import mask as mask_utils
    except Exception as exc:  # pragma: no cover - dependency already present here.
        raise RuntimeError("pycocotools is required to serialize SAM 3 masks.") from exc

    encoded = mask_utils.encode(np.asfortranarray(mask_array.astype(np.uint8)))
    counts = encoded.get("counts")
    if isinstance(counts, bytes):
        encoded["counts"] = counts.decode("utf-8")
    return encoded


def _falcon_prompt_boxes(
    prompt_bboxes: list[dict[str, float]],
    width: int,
    height: int,
    limit: int = 12,
) -> list[list[float]]:
    boxes: list[list[float]] = []
    for bbox in prompt_bboxes[:limit]:
        x0, y0, x1, y1 = bbox_center_wh_norm_to_xyxy(bbox, width, height)
        boxes.append([x0, y0, x1, y1])
    return boxes


def load_sam3_runtime(
    model_id: str = DEFAULT_SAM3_MODEL_ID,
    *,
    allow_experimental_non_cuda: bool = False,
) -> dict[str, Any]:
    torch, device = _torch_device()
    if str(device) != "cuda" and not allow_experimental_non_cuda:
        raise RuntimeError(
            "SAM 3 is disabled on non-CUDA hardware by default. "
            "Pass allow_experimental_non_cuda=True to force a best-effort run."
        )

    try:
        from transformers import Sam3Model, Sam3Processor
    except Exception as exc:  # pragma: no cover - dependency surfaced in UI.
        raise RuntimeError("SAM 3 support requires a recent transformers install.") from exc

    try:
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id)
    except Exception as exc:
        raise RuntimeError(
            "Could not load SAM 3. The checkpoint may require Hugging Face access approval "
            "or additional authentication."
        ) from exc

    model.to(device)
    model.eval()
    return {
        "model_id": model_id,
        "device": str(device),
        "torch": torch,
        "model": model,
        "processor": processor,
        "experimental": str(device) != "cuda",
    }


def run_sam3_inference(
    *,
    runtime: dict[str, Any],
    image: Image.Image,
    query: str,
    prompt_bboxes: list[dict[str, float]],
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> dict[str, Any]:
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]
    width, height = image.size
    prompt_boxes = _falcon_prompt_boxes(prompt_bboxes, width, height)

    processor_kwargs: dict[str, Any] = {
        "images": image,
        "text": query,
        "return_tensors": "pt",
    }
    if prompt_boxes:
        processor_kwargs["input_boxes"] = [prompt_boxes]
        processor_kwargs["input_boxes_labels"] = [[1 for _ in prompt_boxes]]

    inputs = processor(**processor_kwargs)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    target_sizes = inputs.get("original_sizes")
    if hasattr(target_sizes, "tolist"):
        target_sizes = target_sizes.tolist()

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    generation_seconds = time.perf_counter() - start

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=target_sizes,
    )[0]

    detections: list[dict[str, Any]] = []
    bboxes: list[dict[str, float]] = []
    masks_rle: list[dict[str, Any]] = []
    boxes = results.get("boxes") or []
    scores = results.get("scores") or []
    masks = results.get("masks") or []

    for index, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
        if hasattr(box, "tolist"):
            box = box.tolist()
        if hasattr(score, "item"):
            score = score.item()
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask_array = np.asarray(mask).astype(np.uint8)
        if mask_array.ndim > 2:
            mask_array = np.squeeze(mask_array)
        bbox = bbox_xyxy_to_center_wh_norm(float(box[0]), float(box[1]), float(box[2]), float(box[3]), width, height)
        bboxes.append(bbox)
        masks_rle.append(_encode_binary_mask(mask_array))
        detections.append(
            _serialize_detection(
                index=index,
                bbox=bbox,
                label=query,
                score=float(score),
                has_mask=True,
                engine="sam3",
                support_reason="rtdetr prompt boxes" if prompt_boxes else "text prompt",
            )
        )

    summary = (
        f"SAM 3 refined '{query}' into {len(detections)} instance masks "
        f"using {len(prompt_boxes)} RT-DETR prompt boxes selected by Falcon."
    )
    return {
        "engine": "sam3",
        "model_id": runtime["model_id"],
        "device": runtime["device"],
        "decoded_output": summary,
        "detections": detections,
        "bboxes": bboxes,
        "num_masks": len(masks_rle),
        "masks_rle": masks_rle,
        "generation_seconds": generation_seconds,
        "prompt_boxes_count": len(prompt_boxes),
        "experimental": runtime.get("experimental", False),
    }


def _copy_detections(detections: list[dict[str, Any]], *, fallback_engine: str, fallback_label: str) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for detection in detections:
        item = deepcopy(detection)
        item.setdefault("engine", fallback_engine)
        item.setdefault("label", fallback_label)
        item.setdefault("score", None)
        copied.append(item)
    return copied


def _attach_support_annotations(
    *,
    primary_detections: list[dict[str, Any]],
    primary_bboxes: list[dict[str, float]],
    rtdetr_detections: list[dict[str, Any]] | None,
    sam3_detections: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    annotated = _copy_detections(primary_detections, fallback_engine="falcon", fallback_label="query")
    rtdetr_detections = rtdetr_detections or []
    sam3_detections = sam3_detections or []

    for index, detection in enumerate(annotated):
        bbox = primary_bboxes[index] if index < len(primary_bboxes) else {
            "x": detection["center"]["x"],
            "y": detection["center"]["y"],
            "w": detection["width"],
            "h": detection["height"],
        }
        supporting_labels = sorted(
            {
                candidate.get("label")
                for candidate in rtdetr_detections
                if candidate.get("label")
                and intersection_over_union(
                    bbox,
                    {
                        "x": candidate["center"]["x"],
                        "y": candidate["center"]["y"],
                        "w": candidate["width"],
                        "h": candidate["height"],
                    },
                )
                >= 0.25
            }
        )
        sam3_overlap = max(
            (
                intersection_over_union(
                    bbox,
                    {
                        "x": candidate["center"]["x"],
                        "y": candidate["center"]["y"],
                        "w": candidate["width"],
                        "h": candidate["height"],
                    },
                )
                for candidate in sam3_detections
            ),
            default=0.0,
        )
        detection["supporting_labels"] = supporting_labels
        detection["sam3_overlap"] = round(sam3_overlap, 3)
    return annotated


def build_orchestrated_inference(
    *,
    query: str,
    falcon_inference: dict[str, Any],
    falcon_model_id: str,
    falcon_device: str = "mlx",
    falcon_error: str | None = None,
    rtdetr_inference: dict[str, Any] | None = None,
    rtdetr_model_id: str | None = None,
    sam3_inference: dict[str, Any] | None = None,
    sam3_model_id: str | None = None,
    rtdetr_error: str | None = None,
    sam3_error: str | None = None,
) -> dict[str, Any]:
    primary_engine = "falcon"
    primary = falcon_inference
    if sam3_inference and sam3_inference.get("detections"):
        primary_engine = "sam3"
        primary = sam3_inference
    elif rtdetr_inference and rtdetr_inference.get("detections"):
        primary_engine = "rt_detr"
        primary = rtdetr_inference

    primary_detections = _attach_support_annotations(
        primary_detections=primary.get("detections") or [],
        primary_bboxes=primary.get("bboxes") or [],
        rtdetr_detections=(
            None
            if primary_engine == "rt_detr" or not rtdetr_inference
            else rtdetr_inference.get("detections")
        ),
        sam3_detections=(
            None
            if primary_engine == "sam3" or not sam3_inference
            else sam3_inference.get("detections")
        ),
    )

    total_generation = float(falcon_inference.get("generation_seconds") or 0.0)
    if rtdetr_inference:
        total_generation += float(rtdetr_inference.get("generation_seconds") or 0.0)
    if sam3_inference:
        total_generation += float(sam3_inference.get("generation_seconds") or 0.0)

    engines = [
        make_engine_record(
            name="falcon",
            enabled=True,
            status="error" if falcon_error else "ok",
            available=True,
            device=falcon_device,
            model_id=falcon_model_id,
            reason=falcon_error or "Natural-language orchestrator",
            detections_count=len(falcon_inference.get("detections") or []),
            num_masks=int(falcon_inference.get("num_masks") or 0),
            generation_seconds=float(falcon_inference.get("generation_seconds") or 0.0),
        ),
        make_engine_record(
            name="rt_detr",
            enabled=bool(rtdetr_model_id),
            status="ok" if rtdetr_inference else ("error" if rtdetr_error else "skipped"),
            available=rtdetr_inference is not None or rtdetr_error is not None or not bool(rtdetr_model_id),
            device=None if not rtdetr_inference else rtdetr_inference.get("device"),
            model_id=rtdetr_model_id,
            reason=rtdetr_error,
            detections_count=0 if not rtdetr_inference else len(rtdetr_inference.get("detections") or []),
            num_masks=0 if not rtdetr_inference else int(rtdetr_inference.get("num_masks") or 0),
            generation_seconds=None if not rtdetr_inference else float(rtdetr_inference.get("generation_seconds") or 0.0),
        ),
        make_engine_record(
            name="sam3",
            enabled=bool(sam3_model_id),
            status="ok" if sam3_inference else ("error" if sam3_error else "skipped"),
            available=sam3_inference is not None or sam3_error is not None or not bool(sam3_model_id),
            device=None if not sam3_inference else sam3_inference.get("device"),
            model_id=sam3_model_id,
            reason=sam3_error,
            experimental=bool(sam3_inference and sam3_inference.get("experimental")),
            detections_count=0 if not sam3_inference else len(sam3_inference.get("detections") or []),
            num_masks=0 if not sam3_inference else int(sam3_inference.get("num_masks") or 0),
            generation_seconds=None if not sam3_inference else float(sam3_inference.get("generation_seconds") or 0.0),
            prompt_boxes_count=None if not sam3_inference else sam3_inference.get("prompt_boxes_count"),
        ),
    ]

    return {
        "orchestrator": "falcon",
        "query": query,
        "primary_engine": primary_engine,
        "falcon_guidance_count": len(falcon_inference.get("detections") or []),
        "decoded_output": primary.get("decoded_output"),
        "detections": primary_detections,
        "bboxes": primary.get("bboxes") or [],
        "num_masks": int(primary.get("num_masks") or 0),
        "masks_rle": primary.get("masks_rle") or [],
        "generation_seconds": total_generation,
        "engines": engines,
        "engine_outputs": {
            "falcon": falcon_inference,
            "rt_detr": rtdetr_inference,
            "sam3": sam3_inference,
        },
    }
