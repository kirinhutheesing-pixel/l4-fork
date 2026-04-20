#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from perception_orchestrator import (
    DEFAULT_RT_DETR_MODEL_ID,
    DEFAULT_SAM3_MODEL_ID,
    build_orchestrated_inference,
    load_rtdetr_runtime,
    load_sam3_runtime,
    run_rtdetr_inference,
    run_sam3_inference,
)
from falcon_perception import PERCEPTION_MODEL_ID, build_prompt_for_task, load_and_prepare_model
from falcon_perception.data import load_image, stream_samples_from_hf_dataset

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "falcon-pipeline"
DEFAULT_CACHE_DIR = ROOT / ".cache" / "huggingface"
DEFAULT_DEMO_IMAGE = ROOT / "Falcon-Perception-main" / "assets" / "logo.png"
PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Falcon Pipeline on Apple Silicon using the MLX backend. "
            "Pass an image or livestream and a natural-language query to get detections and saved results."
        )
    )
    parser.add_argument("--image", type=str, help="Local image path or URL.")
    parser.add_argument("--stream", type=str, help="Livestream URL or video stream URL. YouTube watch URLs are supported.")
    parser.add_argument("--query", type=str, help="Object or expression to segment or detect.")
    parser.add_argument(
        "--task",
        choices=("segmentation", "detection"),
        default="segmentation",
        help="Whether to return masks or only bounding boxes.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a sample from the PBench dataset if no image is supplied.",
    )
    parser.add_argument(
        "--model-id",
        default=PERCEPTION_MODEL_ID,
        help="Hugging Face model id. Defaults to tiiuae/Falcon-Perception.",
    )
    parser.add_argument(
        "--local-model-dir",
        type=str,
        default=None,
        help="Load model weights from a local Hugging Face export instead of downloading them.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help="MLX dtype to use for inference.",
    )
    parser.add_argument("--min-dim", type=int, default=256, help="Minimum resized image side.")
    parser.add_argument("--max-dim", type=int, default=1024, help="Maximum resized image side.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Generation limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where images and JSON outputs will be written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory for Hugging Face model and Xet cache data.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip saving the overlay image and only write the JSON result.",
    )
    parser.add_argument(
        "--stream-max-samples",
        type=int,
        default=1,
        help="How many frames to sample from a livestream.",
    )
    parser.add_argument(
        "--stream-sample-interval",
        type=float,
        default=5.0,
        help="Seconds to wait between livestream samples.",
    )
    parser.add_argument(
        "--stream-open-timeout",
        type=float,
        default=30.0,
        help="Seconds to keep trying before giving up on opening the livestream.",
    )
    parser.add_argument(
        "--stream-read-timeout",
        type=float,
        default=20.0,
        help="Seconds to keep trying before giving up on reading each livestream frame.",
    )
    parser.add_argument(
        "--enable-rtdetr",
        action="store_true",
        help="Run RT-DETR as the Falcon Pipeline detection backbone.",
    )
    parser.add_argument(
        "--rtdetr-model-id",
        type=str,
        default=DEFAULT_RT_DETR_MODEL_ID,
        help="Hugging Face model id for the RT-DETR stage.",
    )
    parser.add_argument(
        "--rtdetr-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for RT-DETR detections before Falcon alignment filtering.",
    )
    parser.add_argument(
        "--enable-sam3",
        action="store_true",
        help="Run SAM 3 as a refinement stage using Falcon prompt boxes.",
    )
    parser.add_argument(
        "--sam3-model-id",
        type=str,
        default=DEFAULT_SAM3_MODEL_ID,
        help="Hugging Face model id for the SAM 3 refinement stage.",
    )
    parser.add_argument(
        "--sam3-threshold",
        type=float,
        default=0.5,
        help="Instance threshold for SAM 3 post-processing.",
    )
    parser.add_argument(
        "--sam3-mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold for SAM 3 post-processing.",
    )
    parser.add_argument(
        "--allow-experimental-sam3",
        action="store_true",
        help="Allow a best-effort SAM 3 run on non-CUDA hardware.",
    )
    return parser.parse_args()


def configure_huggingface_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    # The Falcon model files are public, and plain HTTP downloads have been
    # more reliable here than the Xet-backed path for first-run setup.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def pair_bbox_entries(raw_entries: list[dict]) -> list[dict]:
    bboxes: list[dict] = []
    current: dict[str, float] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        current.update(entry)
        if all(key in current for key in ("x", "y", "h", "w")):
            bboxes.append(dict(current))
            current = {}
    return bboxes


def decode_rle_mask(mask_rle: dict) -> np.ndarray | None:
    try:
        from pycocotools import mask as mask_utils

        return mask_utils.decode(mask_rle).astype(np.uint8)
    except Exception:
        return None


def render_visualization(
    image: Image.Image,
    bboxes: list[dict],
    masks_rle: list[dict],
    interior_opacity: float = 0.35,
    border_thickness: int = 3,
) -> Image.Image:
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size
    overlay = np.array(rgb_image, dtype=np.float32)

    decoded_masks: list[np.ndarray] = []
    for mask_rle in masks_rle:
        mask = decode_rle_mask(mask_rle)
        if mask is None:
            continue
        if mask.shape != (height, width):
            mask = np.array(Image.fromarray(mask).resize((width, height), Image.NEAREST))
        decoded_masks.append(mask)

    for index, mask in enumerate(decoded_masks):
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        region = mask > 0
        overlay[region] = overlay[region] * (1.0 - interior_opacity) + color * interior_opacity

        from scipy.ndimage import binary_dilation

        border = binary_dilation(region, iterations=border_thickness) & ~region
        overlay[border] = color

    result = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)

    for index, bbox in enumerate(bboxes):
        cx, cy = bbox["x"] * width, bbox["y"] * height
        bw, bh = bbox["w"] * width, bbox["h"] * height
        x0, y0 = cx - bw / 2.0, cy - bh / 2.0
        x1, y1 = cx + bw / 2.0, cy + bh / 2.0
        draw.rectangle([x0, y0, x1, y1], outline=PALETTE[index % len(PALETTE)], width=2)

    return result


def make_slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.strip().lower())
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact[:48] or "result"


def summarize_decoded_output(decoded_text: str) -> str:
    if "<|end_of_image|>" in decoded_text:
        decoded_text = decoded_text.split("<|end_of_image|>", 1)[1]
    while decoded_text.endswith("<|pad|>"):
        decoded_text = decoded_text[: -len("<|pad|>")]
    return decoded_text.strip()


def resolve_static_input(args: argparse.Namespace) -> tuple[Image.Image, str, str]:
    if args.image:
        if not args.query:
            raise SystemExit("--query is required when --image is provided.")
        return load_image(args.image).convert("RGB"), args.query, args.image

    if args.demo:
        sample = stream_samples_from_hf_dataset("tiiuae/PBench", split="level_1")[0]
        sample_query = sample.get("expression") or sample.get("expressions") or "all objects"
        if isinstance(sample_query, list):
            sample_query = ", ".join(str(item) for item in sample_query) if sample_query else "all objects"
        query = args.query or str(sample_query)
        return sample["image"].convert("RGB"), query, "tiiuae/PBench level_1 sample"

    if DEFAULT_DEMO_IMAGE.exists():
        query = args.query or "logo"
        return Image.open(DEFAULT_DEMO_IMAGE).convert("RGB"), query, str(DEFAULT_DEMO_IMAGE)

    raise SystemExit("Provide --image PATH_OR_URL --query TEXT, or pass --demo.")


def is_youtube_url(value: str) -> bool:
    lowered = value.lower()
    return "youtube.com/" in lowered or "youtu.be/" in lowered


def resolve_stream_source(stream: str) -> dict[str, Any]:
    if not is_youtube_url(stream):
        return {"requested_url": stream, "resolved_url": stream, "source_type": "direct"}

    try:
        import yt_dlp
    except ImportError as exc:
        raise SystemExit(
            "yt-dlp is required for YouTube livestream URLs. Install it in the workspace venv first."
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": "best[protocol^=m3u8][vcodec!=none]/best[vcodec!=none]/best",
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(stream, download=False)

    resolved_url = info.get("url")
    if not resolved_url:
        requested_formats = info.get("formats") or []
        for fmt in requested_formats:
            candidate = fmt.get("url")
            if candidate and fmt.get("vcodec") != "none":
                resolved_url = candidate
                break

    if not resolved_url:
        raise SystemExit("Could not resolve a playable media URL from the YouTube livestream.")

    return {
        "requested_url": stream,
        "resolved_url": resolved_url,
        "source_type": "youtube",
        "webpage_url": info.get("webpage_url") or stream,
        "title": info.get("title"),
        "channel": info.get("channel"),
        "is_live": info.get("is_live"),
    }


def open_video_capture(stream_info: dict[str, Any], timeout_seconds: float) -> cv2.VideoCapture:
    deadline = time.time() + max(timeout_seconds, 1.0)
    last_error = "unknown error"
    while time.time() < deadline:
        capture = cv2.VideoCapture(stream_info["resolved_url"], cv2.CAP_FFMPEG)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()
        last_error = f"failed to open {stream_info['resolved_url']}"
        time.sleep(1.0)
    raise SystemExit(f"Could not open livestream within {timeout_seconds:.1f}s: {last_error}")


def read_frame_as_pil(capture: cv2.VideoCapture, timeout_seconds: float) -> Image.Image:
    deadline = time.time() + max(timeout_seconds, 1.0)
    while time.time() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None and frame.size:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        time.sleep(0.25)
    raise SystemExit(f"Could not read a frame from the livestream within {timeout_seconds:.1f}s.")


def run_inference_on_image(
    *,
    engine: Any,
    tokenizer: Any,
    model_args: Any,
    image: Image.Image,
    query: str,
    task: str,
    min_dim: int,
    max_dim: int,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    from falcon_perception.mlx.batch_inference import process_batch_and_generate

    prompt = build_prompt_for_task(query, task)
    batch = process_batch_and_generate(
        tokenizer,
        [(image, prompt)],
        max_length=model_args.max_seq_len,
        min_dimension=min_dim,
        max_dimension=max_dim,
    )

    start = time.perf_counter()
    output_tokens, aux_outputs = engine.generate(
        tokens=batch["tokens"],
        pos_t=batch["pos_t"],
        pos_hw=batch["pos_hw"],
        pixel_values=batch["pixel_values"],
        pixel_mask=batch["pixel_mask"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        task=task,
    )
    generation_seconds = time.perf_counter() - start

    decoded = summarize_decoded_output(
        tokenizer.decode(np.array(output_tokens[0]).tolist(), skip_special_tokens=False)
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
            }
        )

    return {
        "decoded_output": decoded,
        "detections": detections,
        "bboxes": bboxes,
        "num_masks": len(aux.masks_rle),
        "masks_rle": aux.masks_rle,
        "generation_seconds": generation_seconds,
    }


def run_orchestrated_inference_on_image(
    *,
    engine: Any,
    tokenizer: Any,
    model_args: Any,
    image: Image.Image,
    query: str,
    task: str,
    min_dim: int,
    max_dim: int,
    max_new_tokens: int,
    temperature: float,
    falcon_model_id: str,
    enable_rtdetr: bool = False,
    rtdetr_runtime: dict[str, Any] | None = None,
    rtdetr_model_id: str | None = None,
    rtdetr_load_error: str | None = None,
    rtdetr_threshold: float = 0.35,
    enable_sam3: bool = False,
    sam3_runtime: dict[str, Any] | None = None,
    sam3_model_id: str | None = None,
    sam3_load_error: str | None = None,
    sam3_threshold: float = 0.5,
    sam3_mask_threshold: float = 0.5,
) -> dict[str, Any]:
    falcon_inference = run_inference_on_image(
        engine=engine,
        tokenizer=tokenizer,
        model_args=model_args,
        image=image,
        query=query,
        task=task,
        min_dim=min_dim,
        max_dim=max_dim,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    rtdetr_inference: dict[str, Any] | None = None
    rtdetr_error: str | None = rtdetr_load_error
    if enable_rtdetr:
        if rtdetr_runtime is None:
            rtdetr_error = rtdetr_error or "RT-DETR runtime is not loaded."
        else:
            try:
                rtdetr_inference = run_rtdetr_inference(
                    runtime=rtdetr_runtime,
                    image=image,
                    query=query,
                    falcon_bboxes=falcon_inference["bboxes"],
                    threshold=rtdetr_threshold,
                )
            except Exception as exc:
                rtdetr_error = str(exc)

    sam3_inference: dict[str, Any] | None = None
    sam3_error: str | None = sam3_load_error
    if enable_sam3:
        sam3_prompt_bboxes = (
            rtdetr_inference["bboxes"]
            if rtdetr_inference and rtdetr_inference.get("bboxes")
            else falcon_inference["bboxes"]
        )
        if sam3_runtime is None:
            sam3_error = sam3_error or "SAM 3 runtime is not loaded."
        else:
            try:
                sam3_inference = run_sam3_inference(
                    runtime=sam3_runtime,
                    image=image,
                    query=query,
                    prompt_bboxes=sam3_prompt_bboxes,
                    threshold=sam3_threshold,
                    mask_threshold=sam3_mask_threshold,
                )
            except Exception as exc:
                sam3_error = str(exc)

    return build_orchestrated_inference(
        query=query,
        falcon_inference=falcon_inference,
        falcon_model_id=falcon_model_id,
        rtdetr_inference=rtdetr_inference,
        rtdetr_model_id=rtdetr_model_id if enable_rtdetr else None,
        sam3_inference=sam3_inference,
        sam3_model_id=sam3_model_id if enable_sam3 else None,
        rtdetr_error=rtdetr_error,
        sam3_error=sam3_error,
    )


def save_sample_outputs(
    *,
    output_dir: Path,
    stem: str,
    image: Image.Image,
    inference: dict[str, Any],
    skip_visualization: bool,
) -> tuple[Path, Path | None]:
    input_copy_path = output_dir / f"{stem}-input.png"
    image.save(input_copy_path)

    visualization_path: Path | None = None
    if not skip_visualization:
        visualization_path = output_dir / f"{stem}-overlay.png"
        overlay = render_visualization(image, inference["bboxes"], inference["masks_rle"])
        visualization_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(visualization_path)

    return input_copy_path, visualization_path


def validate_args(args: argparse.Namespace) -> None:
    static_inputs = [bool(args.image), bool(args.demo)]
    if args.stream and any(static_inputs):
        raise SystemExit("Use either --stream or a static input (--image/--demo), not both.")
    if args.stream and not args.query:
        raise SystemExit("--query is required when --stream is provided.")
    if args.stream_max_samples < 1:
        raise SystemExit("--stream-max-samples must be at least 1.")


def build_static_result(
    *,
    args: argparse.Namespace,
    query: str,
    input_label: str,
    image: Image.Image,
    inference: dict[str, Any],
    input_copy_path: Path,
    visualization_path: Path | None,
) -> dict[str, Any]:
    return {
        "mode": "image",
        "input": input_label,
        "input_copy": str(input_copy_path),
        "query": query,
        "task": args.task,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "image_size": {"width": image.size[0], "height": image.size[1]},
        "timing_seconds": {"generation": inference["generation_seconds"]},
        "decoded_output": inference["decoded_output"],
        "detections": inference["detections"],
        "num_masks": inference["num_masks"],
        "primary_engine": inference.get("primary_engine", "falcon"),
        "orchestrator": inference.get("orchestrator", "falcon"),
        "engines": inference.get("engines", []),
        "engine_outputs": inference.get("engine_outputs", {}),
        "visualization": None if visualization_path is None else str(visualization_path),
    }


def build_stream_result(
    *,
    args: argparse.Namespace,
    query: str,
    stream_info: dict[str, Any],
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "mode": "stream",
        "input": stream_info["requested_url"],
        "query": query,
        "task": args.task,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "orchestrator": "falcon",
        "enabled_engines": {
            "falcon": True,
            "rt_detr": args.enable_rtdetr,
            "sam3": args.enable_sam3,
        },
        "stream": {
            "source_type": stream_info.get("source_type"),
            "resolved_url": stream_info.get("resolved_url"),
            "webpage_url": stream_info.get("webpage_url"),
            "title": stream_info.get("title"),
            "channel": stream_info.get("channel"),
            "is_live": stream_info.get("is_live"),
        },
        "samples": samples,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    configure_huggingface_cache(args.cache_dir)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Falcon Pipeline...")
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=args.model_id,
        hf_local_dir=args.local_model_dir,
        dtype=args.dtype,
        backend="mlx",
    )

    if args.task == "segmentation" and not model_args.do_segmentation:
        print("Selected model does not support segmentation; switching to detection.")
        args.task = "detection"

    from falcon_perception.mlx.batch_inference import BatchInferenceEngine

    engine = BatchInferenceEngine(model, tokenizer)
    rtdetr_runtime: dict[str, Any] | None = None
    sam3_runtime: dict[str, Any] | None = None
    rtdetr_load_error: str | None = None
    sam3_load_error: str | None = None

    if args.enable_rtdetr:
        print(f"Loading RT-DETR backbone model: {args.rtdetr_model_id}")
        try:
            rtdetr_runtime = load_rtdetr_runtime(args.rtdetr_model_id)
        except Exception as exc:
            rtdetr_load_error = str(exc)
            print(f"Warning: RT-DETR will be skipped for this run: {rtdetr_load_error}")

    if args.enable_sam3:
        print(f"Loading SAM 3 refinement model: {args.sam3_model_id}")
        try:
            sam3_runtime = load_sam3_runtime(
                args.sam3_model_id,
                allow_experimental_non_cuda=args.allow_experimental_sam3,
            )
        except Exception as exc:
            sam3_load_error = str(exc)
            print(f"Warning: SAM 3 will be skipped for this run: {sam3_load_error}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if args.stream:
        query = args.query
        stream_info = resolve_stream_source(args.stream)
        print(f"Resolved stream source: {stream_info.get('title') or stream_info['requested_url']}")
        capture = open_video_capture(stream_info, args.stream_open_timeout)
        samples: list[dict[str, Any]] = []
        try:
            for sample_index in range(args.stream_max_samples):
                image = read_frame_as_pil(capture, args.stream_read_timeout).convert("RGB")
                inference = run_orchestrated_inference_on_image(
                    engine=engine,
                    tokenizer=tokenizer,
                    model_args=model_args,
                    image=image,
                    query=query,
                    task=args.task,
                    min_dim=args.min_dim,
                    max_dim=args.max_dim,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    falcon_model_id=args.model_id,
                    enable_rtdetr=args.enable_rtdetr,
                    rtdetr_runtime=rtdetr_runtime,
                    rtdetr_model_id=args.rtdetr_model_id,
                    rtdetr_load_error=rtdetr_load_error,
                    rtdetr_threshold=args.rtdetr_threshold,
                    enable_sam3=args.enable_sam3,
                    sam3_runtime=sam3_runtime,
                    sam3_model_id=args.sam3_model_id,
                    sam3_load_error=sam3_load_error,
                    sam3_threshold=args.sam3_threshold,
                    sam3_mask_threshold=args.sam3_mask_threshold,
                )
                stem = f"{timestamp}-{make_slug(query)}-sample-{sample_index + 1:02d}"
                input_copy_path, visualization_path = save_sample_outputs(
                    output_dir=output_dir,
                    stem=stem,
                    image=image,
                    inference=inference,
                    skip_visualization=args.skip_visualization,
                )
                samples.append(
                    {
                        "index": sample_index + 1,
                        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "input_copy": str(input_copy_path),
                        "visualization": None if visualization_path is None else str(visualization_path),
                        "image_size": {"width": image.size[0], "height": image.size[1]},
                        "timing_seconds": {"generation": inference["generation_seconds"]},
                        "decoded_output": inference["decoded_output"],
                        "detections": inference["detections"],
                        "num_masks": inference["num_masks"],
                        "primary_engine": inference.get("primary_engine", "falcon"),
                        "engines": inference.get("engines", []),
                        "engine_outputs": inference.get("engine_outputs", {}),
                    }
                )
                print(
                    f"Sample {sample_index + 1}/{args.stream_max_samples}: "
                    f"{len(inference['detections'])} detections via {inference.get('primary_engine', 'falcon')}, "
                    f"{inference['num_masks']} masks, "
                    f"{inference['generation_seconds']:.2f}s"
                )
                if sample_index + 1 < args.stream_max_samples:
                    time.sleep(max(args.stream_sample_interval, 0.0))
        finally:
            capture.release()

        result = build_stream_result(args=args, query=query, stream_info=stream_info, samples=samples)
        json_path = output_dir / f"{timestamp}-{make_slug(query)}-stream.json"
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print()
        print(f"Stream: {stream_info['requested_url']}")
        print(f"Query: {query}")
        print(f"Task: {args.task}")
        print(f"Samples: {len(samples)}")
        print(f"JSON: {json_path}")
        print(f"Cache: {args.cache_dir}")
        return 0

    image, query, input_label = resolve_static_input(args)
    inference = run_orchestrated_inference_on_image(
        engine=engine,
        tokenizer=tokenizer,
        model_args=model_args,
        image=image,
        query=query,
        task=args.task,
        min_dim=args.min_dim,
        max_dim=args.max_dim,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        falcon_model_id=args.model_id,
        enable_rtdetr=args.enable_rtdetr,
        rtdetr_runtime=rtdetr_runtime,
        rtdetr_model_id=args.rtdetr_model_id,
        rtdetr_load_error=rtdetr_load_error,
        rtdetr_threshold=args.rtdetr_threshold,
        enable_sam3=args.enable_sam3,
        sam3_runtime=sam3_runtime,
        sam3_model_id=args.sam3_model_id,
        sam3_load_error=sam3_load_error,
        sam3_threshold=args.sam3_threshold,
        sam3_mask_threshold=args.sam3_mask_threshold,
    )
    stem = f"{timestamp}-{make_slug(query)}"
    input_copy_path, visualization_path = save_sample_outputs(
        output_dir=output_dir,
        stem=stem,
        image=image,
        inference=inference,
        skip_visualization=args.skip_visualization,
    )
    result = build_static_result(
        args=args,
        query=query,
        input_label=input_label,
        image=image,
        inference=inference,
        input_copy_path=input_copy_path,
        visualization_path=visualization_path,
    )
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    print(f"Input: {input_label}")
    print(f"Query: {query}")
    print(f"Task: {args.task}")
    print(f"Primary engine: {inference.get('primary_engine', 'falcon')}")
    print(f"Detections: {len(inference['detections'])}")
    print(f"Masks: {inference['num_masks']}")
    print(f"Generation: {inference['generation_seconds']:.2f}s")
    print(f"JSON: {json_path}")
    if visualization_path is not None:
        print(f"Overlay: {visualization_path}")
    print(f"Input copy: {input_copy_path}")
    print(f"Cache: {args.cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
