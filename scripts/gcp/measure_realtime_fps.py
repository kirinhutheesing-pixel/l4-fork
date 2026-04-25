#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from typing import Any


def fetch_bytes(url: str, timeout: float) -> tuple[int, int, float]:
    start = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = response.read()
        status = int(response.status)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return status, len(payload), elapsed_ms


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Falcon Pipeline realtime /api/frame.jpg delivery.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="Base service URL.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Measurement window.")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout per request.")
    parser.add_argument("--interval", type=float, default=0.0, help="Optional delay between frame requests.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    frame_url = f"{base_url}/api/frame.jpg"
    state_url = f"{base_url}/api/state"
    deadline = time.perf_counter() + max(0.5, args.seconds)
    frame_latencies: list[float] = []
    frame_bytes: list[int] = []
    errors: list[str] = []

    start = time.perf_counter()
    while time.perf_counter() < deadline:
        try:
            status, length, elapsed_ms = fetch_bytes(frame_url, args.timeout)
            if status != 200:
                errors.append(f"frame.jpg returned HTTP {status}")
            else:
                frame_latencies.append(elapsed_ms)
                frame_bytes.append(length)
        except Exception as exc:
            errors.append(str(exc))
        if args.interval > 0:
            time.sleep(args.interval)
    elapsed = max(0.001, time.perf_counter() - start)

    state: dict[str, Any] = {}
    try:
        state = fetch_json(state_url, args.timeout)
    except Exception as exc:
        errors.append(f"state fetch failed: {exc}")

    metrics = state.get("metrics") or {}
    result = state.get("result") or {}
    readiness = state.get("readiness") or {}

    payload = {
        "base_url": base_url,
        "duration_seconds": round(elapsed, 3),
        "frame_requests": len(frame_latencies),
        "observed_frame_response_fps": round(len(frame_latencies) / elapsed, 2),
        "frame_response_ms": {
            "avg": None if not frame_latencies else round(statistics.mean(frame_latencies), 3),
            "p50": percentile(frame_latencies, 50),
            "p95": percentile(frame_latencies, 95),
            "max": None if not frame_latencies else round(max(frame_latencies), 3),
        },
        "frame_bytes": {
            "avg": None if not frame_bytes else round(statistics.mean(frame_bytes), 1),
            "min": None if not frame_bytes else min(frame_bytes),
            "max": None if not frame_bytes else max(frame_bytes),
        },
        "service": {
            "service_state": readiness.get("service_state"),
            "full_pipeline_ready": readiness.get("full_pipeline_ready"),
            "sam3_visual_ready": readiness.get("sam3_visual_ready"),
            "primary_engine": result.get("primary_engine"),
        },
        "runtime_metrics": {
            "capture_fps": metrics.get("capture_fps"),
            "display_frame_fps": metrics.get("display_frame_fps"),
            "frame_response_fps": metrics.get("frame_response_fps"),
            "rtdetr_fps": metrics.get("rtdetr_fps"),
            "sam3_fps": metrics.get("sam3_fps"),
            "falcon_fps": metrics.get("falcon_fps"),
            "overlay_render_ms": metrics.get("overlay_render_ms"),
            "jpeg_encode_ms": metrics.get("jpeg_encode_ms"),
            "rtdetr_generation_ms": metrics.get("rtdetr_generation_ms"),
            "sam3_generation_ms": metrics.get("sam3_generation_ms"),
            "falcon_generation_ms": metrics.get("falcon_generation_ms"),
            "gpu_utilization_percent": metrics.get("gpu_utilization_percent"),
            "gpu_memory_used_mb": metrics.get("gpu_memory_used_mb"),
            "gpu_telemetry_error": metrics.get("gpu_telemetry_error"),
        },
        "errors": errors[:10],
    }
    print(json.dumps(payload, indent=2))
    return 0 if frame_latencies and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
