#!/usr/bin/env python3

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from perception_orchestrator import (
    DEFAULT_RT_DETR_MODEL_ID,
    DEFAULT_SAM3_MODEL_ID,
    inspect_engine_capabilities,
    load_rtdetr_runtime,
    load_sam3_runtime,
)
from falcon_perception import PERCEPTION_MODEL_ID, load_and_prepare_model
from falcon_perception.data import load_image
from falcon_perception.mlx.batch_inference import BatchInferenceEngine
from run_falcon_pipeline import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    configure_huggingface_cache,
    make_slug,
    open_video_capture,
    read_frame_as_pil,
    render_visualization,
    resolve_stream_source,
    run_orchestrated_inference_on_image,
    save_sample_outputs,
)


STREAM_PRESETS = {
    "Jimmy's Deck": {
        "url": "https://www.youtube.com/watch?v=9c1oLjB3wIs",
        "label": "Jimmy's Fish House - Deck",
        "description": "Wide outdoor deck with umbrellas, tables, and waterline activity.",
    },
    "Jimmy's Sunset": {
        "url": "https://www.youtube.com/watch?v=emDyfhDmfUk",
        "label": "Jimmy's Fish House - Sunset",
        "description": "Sunset-facing feed with water, sky, and beach traffic.",
    },
}

QUALITY_PROFILES = {
    "Fast monitor": {
        "min_dim": 256,
        "max_dim": 512,
        "max_new_tokens": 160,
        "dtype": "float16",
        "expected_seconds": 22,
        "description": "Best default for this M1 Air. Prioritizes responsiveness over detail.",
    },
    "Balanced": {
        "min_dim": 256,
        "max_dim": 640,
        "max_new_tokens": 200,
        "dtype": "float16",
        "expected_seconds": 30,
        "description": "Sharper masks and boxes, with a noticeably slower turnaround.",
    },
    "Detail": {
        "min_dim": 320,
        "max_dim": 768,
        "max_new_tokens": 240,
        "dtype": "float16",
        "expected_seconds": 42,
        "description": "Highest fidelity profile. Best for single-frame inspection, not live monitoring.",
    },
}

FAST_MODE_HINT = (
    "This MacBook Air M1 with 8 GB RAM can handle sampled live monitoring, "
    "but not smooth 30 FPS video. Fast Monitor at 512 px is the sweet spot."
)

SAM3_HINT = (
    "SAM 3 is wired in as an optional refinement stage, but on this Mac it remains an "
    "experimental non-CUDA path and may need remote CUDA hardware or checkpoint access."
)

DEFAULTS = {
    "source_mode": "Livestream",
    "query": "umbrellas",
    "task": "segmentation",
    "stream_preset": "Jimmy's Deck",
    "custom_stream_url": "",
    "image_path_or_url": "",
    "quality_profile": "Fast monitor",
    "advanced_mode": False,
    "min_dim": 256,
    "max_dim": 512,
    "max_new_tokens": 160,
    "dtype": "float16",
    "enable_rtdetr": True,
    "enable_sam3": False,
    "allow_experimental_sam3": False,
    "rtdetr_model_id": DEFAULT_RT_DETR_MODEL_ID,
    "sam3_model_id": DEFAULT_SAM3_MODEL_ID,
    "rtdetr_threshold": 0.35,
    "sam3_threshold": 0.5,
    "sam3_mask_threshold": 0.5,
    "save_artifacts": True,
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "sample_count": 3,
    "sample_interval": 1.0,
    "open_timeout": 60.0,
    "read_timeout": 30.0,
}


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg-top: #f7efe2;
          --bg-bottom: #ecf3ef;
          --ink: #1e3435;
          --muted: #567174;
          --line: rgba(31, 66, 67, 0.12);
          --card: rgba(255, 251, 245, 0.86);
          --card-strong: rgba(255, 255, 255, 0.92);
          --teal: #0d7c77;
          --teal-deep: #14535b;
          --sand: #f4d7a5;
          --coral: #ee6b4d;
          --mint: #e5f4ef;
          --shadow: 0 18px 42px rgba(25, 51, 52, 0.10);
          --radius-lg: 24px;
          --radius-md: 18px;
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(240, 172, 111, 0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(21, 124, 120, 0.14), transparent 32%),
            linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
          color: var(--ink);
          font-family: "Avenir Next", "Helvetica Neue", sans-serif;
        }

        h1, h2, h3, [data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2 {
          color: var(--ink);
          font-family: "Iowan Old Style", "Palatino Linotype", serif;
          letter-spacing: -0.02em;
        }

        p, label, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
          color: var(--ink);
        }

        [data-testid="stSidebar"] {
          background: linear-gradient(180deg, rgba(255, 248, 238, 0.96), rgba(238, 245, 242, 0.94));
          border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
          color: var(--ink);
        }

        div[data-testid="stForm"] {
          background: rgba(255, 255, 255, 0.56);
          border: 1px solid rgba(31, 66, 67, 0.08);
          border-radius: 22px;
          padding: 1rem 1rem 0.4rem 1rem;
          box-shadow: var(--shadow);
        }

        div[data-testid="stMetric"] {
          background: var(--card-strong);
          border: 1px solid var(--line);
          border-radius: var(--radius-md);
          padding: 0.95rem 1rem;
          box-shadow: var(--shadow);
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
          background: linear-gradient(135deg, var(--coral), #f18958);
          color: white;
          border: none;
          border-radius: 999px;
          font-weight: 700;
          letter-spacing: 0.01em;
          min-height: 2.9rem;
          box-shadow: 0 12px 30px rgba(238, 107, 77, 0.24);
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
          filter: brightness(1.04);
        }

        [data-testid="stTabs"] button[role="tab"] {
          border-radius: 999px;
          border: 1px solid rgba(13, 124, 119, 0.12);
          padding: 0.45rem 1rem;
          background: rgba(255, 255, 255, 0.65);
          color: var(--ink);
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
          background: linear-gradient(135deg, rgba(13, 124, 119, 0.16), rgba(244, 215, 165, 0.42));
          border-color: rgba(13, 124, 119, 0.22);
        }

        .hero {
          background:
            linear-gradient(140deg, rgba(255, 255, 255, 0.78), rgba(232, 246, 243, 0.86)),
            radial-gradient(circle at top right, rgba(238, 107, 77, 0.12), transparent 30%);
          border: 1px solid rgba(31, 66, 67, 0.10);
          border-radius: 28px;
          padding: 1.4rem 1.5rem 1.25rem 1.5rem;
          margin-bottom: 1rem;
          box-shadow: var(--shadow);
        }

        .eyebrow {
          text-transform: uppercase;
          font-size: 0.76rem;
          font-weight: 700;
          letter-spacing: 0.16em;
          color: var(--teal);
          margin-bottom: 0.55rem;
        }

        .hero h1 {
          font-size: 3rem;
          line-height: 0.95;
          margin: 0 0 0.55rem 0;
        }

        .hero p {
          margin: 0;
          max-width: 52rem;
          color: var(--muted);
          font-size: 1.02rem;
        }

        .pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.55rem;
          margin-top: 1rem;
        }

        .pill {
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
          padding: 0.45rem 0.8rem;
          border-radius: 999px;
          background: rgba(13, 124, 119, 0.08);
          border: 1px solid rgba(13, 124, 119, 0.10);
          color: var(--teal-deep);
          font-size: 0.88rem;
          font-weight: 600;
        }

        .banner {
          border-radius: 22px;
          padding: 0.9rem 1rem;
          margin: 0.35rem 0 1rem 0;
          border: 1px solid transparent;
          box-shadow: var(--shadow);
        }

        .banner strong {
          display: block;
          margin-bottom: 0.18rem;
        }

        .banner-info {
          background: rgba(13, 124, 119, 0.09);
          border-color: rgba(13, 124, 119, 0.14);
          color: var(--teal-deep);
        }

        .banner-success {
          background: rgba(114, 185, 122, 0.12);
          border-color: rgba(114, 185, 122, 0.18);
          color: #234d2a;
        }

        .banner-warning {
          background: rgba(240, 172, 111, 0.18);
          border-color: rgba(240, 172, 111, 0.20);
          color: #7b4e0d;
        }

        .banner-error {
          background: rgba(238, 107, 77, 0.13);
          border-color: rgba(238, 107, 77, 0.18);
          color: #7a2f1d;
        }

        .micro-card {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 18px;
          padding: 0.9rem 1rem;
          margin-bottom: 0.8rem;
          box-shadow: var(--shadow);
        }

        .micro-card h4 {
          margin: 0 0 0.25rem 0;
          font-family: "Avenir Next", "Helvetica Neue", sans-serif;
          font-size: 0.95rem;
          letter-spacing: 0.02em;
          text-transform: uppercase;
          color: var(--teal);
        }

        .micro-card p {
          margin: 0;
          color: var(--muted);
          font-size: 0.92rem;
        }

        .meta-list {
          display: grid;
          gap: 0.7rem;
        }

        .meta-item {
          background: rgba(255, 255, 255, 0.74);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 0.8rem 0.9rem;
        }

        .meta-item small {
          display: block;
          color: var(--muted);
          text-transform: uppercase;
          font-size: 0.7rem;
          letter-spacing: 0.12em;
          margin-bottom: 0.18rem;
        }

        .meta-item strong {
          color: var(--ink);
          font-size: 0.98rem;
        }

        .empty-state {
          background: rgba(255, 255, 255, 0.74);
          border: 1px solid var(--line);
          border-radius: 24px;
          padding: 1.2rem 1.25rem;
          box-shadow: var(--shadow);
        }

        .section-note {
          color: var(--muted);
          font-size: 0.92rem;
          margin-top: -0.35rem;
          margin-bottom: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    if "control_defaults" not in st.session_state:
        st.session_state["control_defaults"] = dict(DEFAULTS)
    if "latest_run" not in st.session_state:
        st.session_state["latest_run"] = None
    if "run_archive" not in st.session_state:
        st.session_state["run_archive"] = []


def load_image_from_app_input(uploaded_file, image_path_or_url: str) -> tuple[Image.Image, str]:
    if uploaded_file is not None:
        return Image.open(uploaded_file).convert("RGB"), uploaded_file.name
    if image_path_or_url.strip():
        return load_image(image_path_or_url.strip()).convert("RGB"), image_path_or_url.strip()
    raise ValueError("Provide an uploaded image or a local path / URL.")


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.0f}s"


def title_case_engine(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").upper() if name.lower() == "sam3" else name.replace("_", " ").replace("-", " ").title()


def format_score(value: Any) -> str:
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def estimate_runtime(profile_name: str, sample_count: int, sample_interval: float, source_mode: str) -> str:
    profile = QUALITY_PROFILES[profile_name]
    expected = profile["expected_seconds"]
    samples = 1 if source_mode == "Image" else sample_count
    total_seconds = expected * samples + max(samples - 1, 0) * sample_interval
    if total_seconds < 60:
        return f"about {int(round(total_seconds))} seconds"
    minutes, remainder = divmod(total_seconds, 60)
    if remainder < 5:
        return f"about {int(round(minutes))} minutes"
    return f"about {int(minutes)}m {int(remainder)}s"


def resolve_stream_input(preset_name: str, custom_stream_url: str) -> tuple[str, str]:
    custom_value = custom_stream_url.strip()
    if custom_value:
        return custom_value, "Custom livestream"
    preset = STREAM_PRESETS[preset_name]
    return preset["url"], preset["label"]


def escape_text(value: Any) -> str:
    return html.escape(str(value))


def flatten_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det in detections:
        center = det.get("center") or {}
        width = float(det.get("width") or 0.0)
        height = float(det.get("height") or 0.0)
        supporting_labels = det.get("supporting_labels") or []
        rows.append(
            {
                "index": int(det.get("index", 0)),
                "engine": str(det.get("engine") or "falcon"),
                "label": str(det.get("label") or "query"),
                "score": format_score(det.get("score")),
                "center_x": round(float(center.get("x", 0.0)), 4),
                "center_y": round(float(center.get("y", 0.0)), 4),
                "width": round(width, 4),
                "height": round(height, 4),
                "area": round(width * height, 4),
                "mask": "Yes" if det.get("has_mask") else "No",
                "rt_detr_support": ", ".join(str(item) for item in supporting_labels) if supporting_labels else "—",
                "sam3_overlap": det.get("sam3_overlap", "—"),
            }
        )
    return rows


def build_history_frame(samples: list[dict[str, Any]]) -> pd.DataFrame:
    if not samples:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Sample": sample["index"],
                "Detections": sample["detections_count"],
                "Masks": sample["num_masks"],
                "Generation (s)": round(sample["generation_seconds"], 2),
                "Primary engine": sample.get("primary_engine", "falcon"),
            }
            for sample in samples
        ]
    )


def history_line_chart(dataframe: pd.DataFrame, y_field: str, color: str, title: str) -> alt.Chart:
    return (
        alt.Chart(dataframe, title=title)
        .mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=3, color=color)
        .encode(
            x=alt.X("Sample:O", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y_field}:Q"),
            tooltip=["Sample:O", f"{y_field}:Q"],
        )
        .properties(height=260)
    )


def render_banner(slot, tone: str, title: str, body: str) -> None:
    slot.markdown(
        (
            f'<div class="banner banner-{tone}">' 
            f"<strong>{escape_text(title)}</strong>"
            f"<span>{escape_text(body)}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">Falcon Pipeline Ops</div>
          <h1>Live Viewer</h1>
          <p>
            A production-style monitoring console for livestream and still-image perception on Apple Silicon.
            Falcon stays in charge as the orchestrator, RT-DETR serves as the detection backbone, and SAM 3 refines the final segmentation when the hardware allows it.
          </p>
          <div class="pill-row">
            <span class="pill">Falcon orchestrator</span>
            <span class="pill">RT-DETR backbone</span>
            <span class="pill">SAM 3 refinement path</span>
            <span class="pill">Optimized for M1 / 8 GB</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <h3 style="margin-top:0;">Ready for a monitoring run</h3>
          <p class="section-note">
            Configure the source in the left rail, then submit a run. The dashboard will stream sample-by-sample
            results here with frame previews, overlays, detections, and timing analytics.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_one, col_two, col_three = st.columns(3)
    with col_one:
        st.markdown(
            """
            <div class="micro-card">
              <h4>Fastest setup</h4>
              <p>Use <strong>Jimmy's Deck</strong> with the <strong>Fast monitor</strong> profile for the best balance on this Mac.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_two:
        st.markdown(
            """
            <div class="micro-card">
              <h4>Best queries</h4>
              <p>Try phrases like <strong>green umbrellas</strong>, <strong>boats near the horizon</strong>, or <strong>tables by the railing</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_three:
        st.markdown(
            f"""
            <div class="micro-card">
              <h4>Pipeline reality</h4>
              <p>{escape_text(SAM3_HINT)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_engine_cards(engines: list[dict[str, Any]]) -> None:
    if not engines:
        return
    columns = st.columns(len(engines))
    for column, engine in zip(columns, engines):
        status = str(engine.get("status") or "unknown").upper()
        detections = int(engine.get("detections_count") or 0)
        masks = int(engine.get("num_masks") or 0)
        device = engine.get("device") or "—"
        note = engine.get("reason") or (
            f"{detections} detections · {masks} masks"
            if engine.get("status") == "ok"
            else "Enabled for this session."
        )
        title = title_case_engine(str(engine.get("name") or "engine"))
        with column:
            st.markdown(
                f"""
                <div class="meta-item">
                  <small>{escape_text(title)}</small>
                  <strong>{escape_text(status)}</strong>
                  <div style="margin-top:0.25rem; color:#567174; font-size:0.88rem;">
                    {escape_text(device)} · {escape_text(note)}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_run_dashboard(run_record: dict[str, Any], archive: list[dict[str, Any]]) -> None:
    samples = run_record["samples"]
    latest_sample = samples[-1]
    sample_frame = build_history_frame(samples)
    detection_frame = pd.DataFrame(latest_sample["detection_rows"])

    detections_series = [sample["detections_count"] for sample in samples]
    generation_series = [sample["generation_seconds"] for sample in samples]
    average_generation = mean(generation_series)
    best_generation = min(generation_series)
    requested_samples = run_record.get("requested_samples", len(samples))
    sample_label = f"{latest_sample['index']}/{requested_samples}"
    latest_engines = latest_sample.get("engines") or run_record.get("engines") or []

    st.markdown(
        f"""
        <div class="micro-card" style="margin-bottom: 1rem;">
          <h4>Current session</h4>
          <p>
            <strong>{escape_text(run_record['source_display'])}</strong> · Query <strong>{escape_text(run_record['query'])}</strong>
            · Task <strong>{escape_text(run_record['task'])}</strong> · Profile <strong>{escape_text(run_record['quality_profile'])}</strong>
            · Primary engine <strong>{escape_text(title_case_engine(run_record.get('primary_engine', 'falcon')))}</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_one, metric_two, metric_three, metric_four, metric_five = st.columns(5)
    metric_one.metric("Current detections", latest_sample["detections_count"])
    metric_two.metric("Current masks", latest_sample["num_masks"])
    metric_three.metric("Latest generation", format_seconds(latest_sample["generation_seconds"]))
    metric_four.metric("Average generation", format_seconds(average_generation))
    metric_five.metric("Samples complete", sample_label)
    render_engine_cards(latest_engines)

    tab_view, tab_detections, tab_notes, tab_history = st.tabs(
        ["Live View", "Detections", "Operator Notes", "History"]
    )

    with tab_view:
        preview_left, preview_right = st.columns(2)
        with preview_left:
            st.subheader("Raw frame")
            st.image(run_record["current_image"], width="stretch")
        with preview_right:
            st.subheader(f"{title_case_engine(run_record.get('primary_engine', 'falcon'))} overlay")
            st.image(run_record["current_overlay"], width="stretch")

        artifact_left, artifact_right, artifact_third = st.columns(3)
        artifact_left.markdown(
            f"""
            <div class="meta-item">
              <small>Captured at</small>
              <strong>{escape_text(latest_sample['captured_at'])}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        artifact_right.markdown(
            f"""
            <div class="meta-item">
              <small>Saved JSON</small>
              <strong>{escape_text(latest_sample['save_paths']['json'] or 'Not saved')}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        artifact_third.markdown(
            f"""
            <div class="meta-item">
              <small>Overlay image</small>
              <strong>{escape_text(latest_sample['save_paths']['overlay'] or 'Not saved')}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_detections:
        table_col, insights_col = st.columns([1.9, 1.1])
        with table_col:
            st.subheader("Detection table")
            st.caption("Normalized coordinates and dimensions for the latest sample.")
            st.dataframe(detection_frame, width="stretch", hide_index=True)

        with insights_col:
            largest_area = max((row["area"] for row in latest_sample["detection_rows"]), default=0.0)
            st.markdown(
                f"""
                <div class="meta-list">
                  <div class="meta-item">
                    <small>Largest box area</small>
                    <strong>{largest_area:.4f}</strong>
                  </div>
                  <div class="meta-item">
                    <small>Detection spread</small>
                    <strong>{max(detections_series) - min(detections_series)}</strong>
                  </div>
                  <div class="meta-item">
                    <small>Best generation</small>
                    <strong>{format_seconds(best_generation)}</strong>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_notes:
        note_left, note_right = st.columns([1.1, 1.3])
        with note_left:
            st.subheader("Run metadata")
            stream_info = run_record.get("stream_info") or {}
            metadata_items = [
                ("Source", run_record["source_display"]),
                ("Source mode", run_record["mode"]),
                ("Query", run_record["query"]),
                ("Task", run_record["task"]),
                ("Orchestrator", title_case_engine(run_record.get("orchestrator", "falcon"))),
                ("Primary engine", title_case_engine(run_record.get("primary_engine", "falcon"))),
                ("Quality profile", run_record["quality_profile"]),
                (
                    "Effective runtime",
                    (
                        f"min {run_record['effective_settings']['min_dim']} · "
                        f"max {run_record['effective_settings']['max_dim']} · "
                        f"tokens {run_record['effective_settings']['max_new_tokens']} · "
                        f"dtype {run_record['effective_settings']['dtype']}"
                    ),
                ),
            ]
            if stream_info:
                metadata_items.extend(
                    [
                        ("Stream title", stream_info.get("title") or "Unknown"),
                        ("Channel", stream_info.get("channel") or "Unknown"),
                        ("Live flag", "Yes" if stream_info.get("is_live") else "Unknown"),
                    ]
                )
            for label, value in metadata_items:
                st.markdown(
                    f"""
                    <div class="meta-item">
                      <small>{escape_text(label)}</small>
                      <strong>{escape_text(value)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if latest_engines:
                st.subheader("Engine trace")
                render_engine_cards(latest_engines)

        with note_right:
            st.subheader("Decoded model output")
            st.code(latest_sample["decoded_output"], language="text")

    with tab_history:
        st.subheader("Performance over samples")
        if len(sample_frame) > 1:
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.altair_chart(
                    history_line_chart(sample_frame, "Detections", "#0d7c77", "Detections per sample"),
                    width="stretch",
                )
            with chart_right:
                st.altair_chart(
                    history_line_chart(sample_frame, "Generation (s)", "#ee6b4d", "Generation time per sample"),
                    width="stretch",
                )
        else:
            st.info("Run more than one sample to unlock trend charts.")

        st.dataframe(sample_frame, width="stretch", hide_index=True)

        if archive:
            st.subheader("Recent sessions")
            archive_frame = pd.DataFrame(archive)
            st.dataframe(archive_frame, width="stretch", hide_index=True)


@st.cache_resource(show_spinner=False)
def load_runtime(model_id: str, dtype: str) -> tuple[Any, Any, Any]:
    configure_huggingface_cache(DEFAULT_CACHE_DIR)
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=model_id,
        dtype=dtype,
        backend="mlx",
    )
    engine = BatchInferenceEngine(model, tokenizer)
    return engine, tokenizer, model_args


@st.cache_resource(show_spinner=False)
def load_cached_rtdetr_runtime(model_id: str) -> dict[str, Any]:
    configure_huggingface_cache(DEFAULT_CACHE_DIR)
    return load_rtdetr_runtime(model_id)


@st.cache_resource(show_spinner=False)
def load_cached_sam3_runtime(model_id: str, allow_experimental_non_cuda: bool) -> dict[str, Any]:
    configure_huggingface_cache(DEFAULT_CACHE_DIR)
    return load_sam3_runtime(
        model_id,
        allow_experimental_non_cuda=allow_experimental_non_cuda,
    )


def save_app_sample(
    *,
    output_dir: Path,
    stem: str,
    image: Image.Image,
    inference: dict[str, Any],
    query: str,
    task: str,
    source_label: str,
    save_artifacts: bool,
) -> dict[str, str | None]:
    if not save_artifacts:
        return {"input_copy": None, "overlay": None, "json": None}

    input_copy_path, overlay_path = save_sample_outputs(
        output_dir=output_dir,
        stem=stem,
        image=image,
        inference=inference,
        skip_visualization=False,
    )
    json_path = output_dir / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "source": source_label,
                "query": query,
                "task": task,
                "decoded_output": inference["decoded_output"],
                "detections": inference["detections"],
                "num_masks": inference["num_masks"],
                "primary_engine": inference.get("primary_engine", "falcon"),
                "orchestrator": inference.get("orchestrator", "falcon"),
                "engines": inference.get("engines", []),
                "engine_outputs": inference.get("engine_outputs", {}),
                "timing_seconds": {"generation": inference["generation_seconds"]},
                "input_copy": str(input_copy_path),
                "visualization": None if overlay_path is None else str(overlay_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "input_copy": str(input_copy_path),
        "overlay": None if overlay_path is None else str(overlay_path),
        "json": str(json_path),
    }


def archive_entry_for_run(run_record: dict[str, Any]) -> dict[str, Any]:
    latest_sample = run_record["samples"][-1]
    generation_values = [sample["generation_seconds"] for sample in run_record["samples"]]
    return {
        "Started": run_record["started_at"],
        "Source": run_record["source_display"],
        "Query": run_record["query"],
        "Task": run_record["task"],
        "Primary": title_case_engine(run_record.get("primary_engine", "falcon")),
        "Profile": run_record["quality_profile"],
        "Samples": len(run_record["samples"]),
        "Final detections": latest_sample["detections_count"],
        "Avg generation": round(mean(generation_values), 2),
    }


def main() -> None:
    st.set_page_config(page_title="Falcon Pipeline Live Viewer", layout="wide")
    inject_styles()
    init_state()
    render_hero()

    defaults = st.session_state["control_defaults"]
    latest_run = st.session_state["latest_run"]
    capability_snapshot = inspect_engine_capabilities(
        allow_experimental_sam3=bool(defaults["allow_experimental_sam3"])
    )

    status_box = st.empty()
    progress_box = st.empty()
    dashboard_box = st.empty()

    with st.sidebar:
        st.markdown(
            """
            <div class="micro-card">
              <h4>Mission control</h4>
              <p>Configure the feed, choose a quality profile, and launch a deliberate run without live-edit reruns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("falcon_control_form", clear_on_submit=False):
            source_mode = st.radio(
                "Source mode",
                ["Livestream", "Image"],
                index=0 if defaults["source_mode"] == "Livestream" else 1,
                horizontal=True,
            )
            query = st.text_input(
                "Natural-language query",
                value=defaults["query"],
                placeholder="green umbrellas, people near the railing, boats on the water",
            )
            task = st.selectbox(
                "Task",
                ["segmentation", "detection"],
                index=0 if defaults["task"] == "segmentation" else 1,
            )
            with st.expander("Orchestration stages", expanded=True):
                enable_rtdetr = st.toggle(
                    "Enable RT-DETR backbone",
                    value=bool(defaults["enable_rtdetr"]),
                )
                enable_sam3 = st.toggle(
                    "Enable SAM 3 refinement",
                    value=bool(defaults["enable_sam3"]),
                )
                allow_experimental_sam3 = st.toggle(
                    "Allow experimental SAM 3 on non-CUDA hardware",
                    value=bool(defaults["allow_experimental_sam3"]),
                    disabled=not enable_sam3,
                )
                stage_capabilities = inspect_engine_capabilities(
                    allow_experimental_sam3=allow_experimental_sam3
                )
                rtdetr_capability = stage_capabilities["rt_detr"]
                sam3_capability = stage_capabilities["sam3"]
                st.caption(
                    "RT-DETR: "
                    + (
                        f"ready as the backbone on {rtdetr_capability.get('device') or 'unknown'}"
                        if rtdetr_capability["available"]
                        else str(rtdetr_capability["reason"])
                    )
                )
                st.caption(
                    "SAM 3: "
                    + (
                        f"ready on {sam3_capability.get('device') or 'unknown'}"
                        if sam3_capability["available"]
                        else str(sam3_capability["reason"])
                    )
                )

            if source_mode == "Livestream":
                stream_preset_names = list(STREAM_PRESETS)
                preset_index = stream_preset_names.index(defaults["stream_preset"])
                stream_preset = st.selectbox("Livestream preset", stream_preset_names, index=preset_index)
                st.caption(STREAM_PRESETS[stream_preset]["description"])
                custom_stream_url = st.text_input(
                    "Custom livestream URL",
                    value=defaults["custom_stream_url"],
                    placeholder="Optional. Leave blank to use the preset feed.",
                )
                image_path_or_url = ""
                uploaded_file = None
                sample_count = int(
                    st.slider("Samples per run", min_value=1, max_value=12, value=int(defaults["sample_count"]))
                )
                sample_interval = float(
                    st.slider(
                        "Seconds between samples",
                        min_value=0.0,
                        max_value=30.0,
                        value=float(defaults["sample_interval"]),
                        step=0.5,
                    )
                )
            else:
                stream_preset = defaults["stream_preset"]
                custom_stream_url = ""
                image_path_or_url = st.text_input(
                    "Image path or URL",
                    value=defaults["image_path_or_url"],
                    placeholder="/absolute/path/to/image.jpg or https://...",
                )
                uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
                sample_count = 1
                sample_interval = 0.0

            quality_names = list(QUALITY_PROFILES)
            quality_index = quality_names.index(defaults["quality_profile"])
            quality_profile = st.selectbox("Quality profile", quality_names, index=quality_index)
            profile = QUALITY_PROFILES[quality_profile]
            st.caption(profile["description"])
            st.markdown(
                f"""
                <div class="micro-card">
                  <h4>Expected wall-clock</h4>
                  <p>{escape_text(estimate_runtime(quality_profile, sample_count, sample_interval, source_mode))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            advanced_mode = st.toggle("Customize runtime controls", value=bool(defaults["advanced_mode"]))
            with st.expander("Advanced runtime controls", expanded=advanced_mode):
                min_dim = int(
                    st.number_input(
                        "Min dimension",
                        min_value=128,
                        max_value=1024,
                        value=int(defaults["min_dim"]),
                        step=32,
                        disabled=not advanced_mode,
                    )
                )
                max_dim = int(
                    st.number_input(
                        "Max dimension",
                        min_value=256,
                        max_value=1024,
                        value=int(defaults["max_dim"]),
                        step=32,
                        disabled=not advanced_mode,
                    )
                )
                max_new_tokens = int(
                    st.number_input(
                        "Max new tokens",
                        min_value=32,
                        max_value=512,
                        value=int(defaults["max_new_tokens"]),
                        step=16,
                        disabled=not advanced_mode,
                    )
                )
                dtype = st.selectbox(
                    "MLX dtype",
                    ["float16", "bfloat16", "float32"],
                    index=["float16", "bfloat16", "float32"].index(defaults["dtype"]),
                    disabled=not advanced_mode,
                )
                open_timeout = float(
                    st.number_input(
                        "Open timeout (s)",
                        min_value=5.0,
                        max_value=120.0,
                        value=float(defaults["open_timeout"]),
                        step=5.0,
                        disabled=not advanced_mode or source_mode != "Livestream",
                    )
                )
                read_timeout = float(
                    st.number_input(
                        "Read timeout (s)",
                        min_value=5.0,
                        max_value=120.0,
                        value=float(defaults["read_timeout"]),
                        step=5.0,
                        disabled=not advanced_mode or source_mode != "Livestream",
                    )
                )
                rtdetr_model_id = st.text_input(
                    "RT-DETR model id",
                    value=str(defaults["rtdetr_model_id"]),
                    disabled=not advanced_mode,
                )
                rtdetr_threshold = float(
                    st.number_input(
                        "RT-DETR threshold",
                        min_value=0.05,
                        max_value=0.95,
                        value=float(defaults["rtdetr_threshold"]),
                        step=0.05,
                        disabled=not advanced_mode or not enable_rtdetr,
                    )
                )
                sam3_model_id = st.text_input(
                    "SAM 3 model id",
                    value=str(defaults["sam3_model_id"]),
                    disabled=not advanced_mode,
                )
                sam3_threshold = float(
                    st.number_input(
                        "SAM 3 instance threshold",
                        min_value=0.05,
                        max_value=0.95,
                        value=float(defaults["sam3_threshold"]),
                        step=0.05,
                        disabled=not advanced_mode or not enable_sam3,
                    )
                )
                sam3_mask_threshold = float(
                    st.number_input(
                        "SAM 3 mask threshold",
                        min_value=0.05,
                        max_value=0.95,
                        value=float(defaults["sam3_mask_threshold"]),
                        step=0.05,
                        disabled=not advanced_mode or not enable_sam3,
                    )
                )
            if not advanced_mode:
                rtdetr_model_id = str(defaults["rtdetr_model_id"])
                sam3_model_id = str(defaults["sam3_model_id"])
                rtdetr_threshold = float(defaults["rtdetr_threshold"])
                sam3_threshold = float(defaults["sam3_threshold"])
                sam3_mask_threshold = float(defaults["sam3_mask_threshold"])

            save_artifacts = st.toggle("Save artifacts to outputs/", value=bool(defaults["save_artifacts"]))
            output_dir = st.text_input("Output directory", value=defaults["output_dir"])
            run_button = st.form_submit_button(
                "Run live monitoring" if source_mode == "Livestream" else "Analyze image",
                width="stretch",
                type="primary",
            )

        st.markdown(
            f"""
            <div class="micro-card">
              <h4>Hardware note</h4>
              <p>{escape_text(FAST_MODE_HINT)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="micro-card">
              <h4>Pipeline note</h4>
              <p>{escape_text(capability_snapshot['sam3']['reason'] or SAM3_HINT)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if advanced_mode:
        effective_settings = {
            "min_dim": min_dim,
            "max_dim": max_dim,
            "max_new_tokens": max_new_tokens,
            "dtype": dtype,
            "open_timeout": open_timeout,
            "read_timeout": read_timeout,
        }
    else:
        effective_settings = {
            "min_dim": profile["min_dim"],
            "max_dim": profile["max_dim"],
            "max_new_tokens": profile["max_new_tokens"],
            "dtype": profile["dtype"],
            "open_timeout": float(defaults["open_timeout"]),
            "read_timeout": float(defaults["read_timeout"]),
        }

    if not run_button:
        if latest_run:
            render_banner(
                status_box,
                "success",
                "Viewer ready",
                "Showing the most recent successful run. Submit a new session from Mission Control to refresh it.",
            )
            with dashboard_box.container():
                render_run_dashboard(latest_run, st.session_state["run_archive"])
        else:
            render_banner(
                status_box,
                "info",
                "Waiting for a run",
                "Choose a source, write a vision query, and launch a session to populate the dashboard.",
            )
            with dashboard_box.container():
                render_empty_state()
        return

    st.session_state["control_defaults"] = {
        "source_mode": source_mode,
        "query": query,
        "task": task,
        "enable_rtdetr": enable_rtdetr,
        "enable_sam3": enable_sam3,
        "allow_experimental_sam3": allow_experimental_sam3,
        "rtdetr_model_id": rtdetr_model_id,
        "sam3_model_id": sam3_model_id,
        "rtdetr_threshold": rtdetr_threshold,
        "sam3_threshold": sam3_threshold,
        "sam3_mask_threshold": sam3_mask_threshold,
        "stream_preset": stream_preset,
        "custom_stream_url": custom_stream_url,
        "image_path_or_url": image_path_or_url,
        "quality_profile": quality_profile,
        "advanced_mode": advanced_mode,
        "min_dim": effective_settings["min_dim"],
        "max_dim": effective_settings["max_dim"],
        "max_new_tokens": effective_settings["max_new_tokens"],
        "dtype": effective_settings["dtype"],
        "save_artifacts": save_artifacts,
        "output_dir": output_dir,
        "sample_count": sample_count,
        "sample_interval": sample_interval,
        "open_timeout": effective_settings["open_timeout"],
        "read_timeout": effective_settings["read_timeout"],
    }

    if not query.strip():
        render_banner(status_box, "error", "Missing query", "Enter a natural-language target before launching a run.")
        with dashboard_box.container():
            render_empty_state()
        return

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        render_banner(
            status_box,
            "info",
            "Loading runtime",
            f"Starting Falcon Pipeline with the {quality_profile} profile on the local MLX backend.",
        )
        with st.spinner("Loading Falcon Pipeline runtime..."):
            engine, tokenizer, model_args = load_runtime(PERCEPTION_MODEL_ID, effective_settings["dtype"])
    except Exception as exc:
        render_banner(status_box, "error", "Runtime load failed", str(exc))
        with dashboard_box.container():
            render_empty_state()
        return

    rtdetr_runtime: dict[str, Any] | None = None
    sam3_runtime: dict[str, Any] | None = None
    rtdetr_load_error: str | None = None
    sam3_load_error: str | None = None
    pipeline_notes: list[str] = []

    if enable_rtdetr:
        try:
            rtdetr_runtime = load_cached_rtdetr_runtime(rtdetr_model_id)
            pipeline_notes.append(f"RT-DETR ready on {rtdetr_runtime['device']}.")
        except Exception as exc:
            rtdetr_load_error = str(exc)
            pipeline_notes.append(f"RT-DETR skipped: {rtdetr_load_error}")

    if enable_sam3:
        try:
            sam3_runtime = load_cached_sam3_runtime(
                sam3_model_id,
                allow_experimental_sam3,
            )
            experimental_suffix = " (experimental)" if sam3_runtime.get("experimental") else ""
            pipeline_notes.append(f"SAM 3 ready on {sam3_runtime['device']}{experimental_suffix}.")
        except Exception as exc:
            sam3_load_error = str(exc)
            pipeline_notes.append(f"SAM 3 skipped: {sam3_load_error}")

    if task == "segmentation" and not model_args.do_segmentation:
        render_banner(
            status_box,
            "warning",
            "Segmentation unavailable",
            "The current model variant does not expose segmentation heads, so the run switched to detection mode.",
        )
        task = "detection"

    if pipeline_notes:
        render_banner(
            status_box,
            "info",
            "Pipeline ready",
            " ".join(pipeline_notes),
        )

    run_record: dict[str, Any] = {
        "mode": source_mode,
        "query": query.strip(),
        "task": task,
        "orchestrator": "falcon",
        "primary_engine": "falcon",
        "quality_profile": quality_profile,
        "effective_settings": effective_settings,
        "enabled_engines": {
            "falcon": True,
            "rt_detr": enable_rtdetr,
            "sam3": enable_sam3,
        },
        "pipeline_settings": {
            "enable_rtdetr": enable_rtdetr,
            "enable_sam3": enable_sam3,
            "allow_experimental_sam3": allow_experimental_sam3,
            "rtdetr_model_id": rtdetr_model_id,
            "sam3_model_id": sam3_model_id,
            "rtdetr_threshold": rtdetr_threshold,
            "sam3_threshold": sam3_threshold,
            "sam3_mask_threshold": sam3_mask_threshold,
        },
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_samples": sample_count,
        "samples": [],
        "stream_info": None,
        "source_display": "",
        "current_image": None,
        "current_overlay": None,
        "engines": [],
    }
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    try:
        if source_mode == "Image":
            image, source_label = load_image_from_app_input(uploaded_file, image_path_or_url)
            run_record["source_display"] = source_label
            progress_box.progress(0.15, text="Running image inference...")

            inference = run_orchestrated_inference_on_image(
                engine=engine,
                tokenizer=tokenizer,
                model_args=model_args,
                image=image,
                query=query.strip(),
                task=task,
                min_dim=effective_settings["min_dim"],
                max_dim=effective_settings["max_dim"],
                max_new_tokens=effective_settings["max_new_tokens"],
                temperature=0.0,
                falcon_model_id=PERCEPTION_MODEL_ID,
                enable_rtdetr=enable_rtdetr,
                rtdetr_runtime=rtdetr_runtime,
                rtdetr_model_id=rtdetr_model_id,
                rtdetr_load_error=rtdetr_load_error,
                rtdetr_threshold=rtdetr_threshold,
                enable_sam3=enable_sam3,
                sam3_runtime=sam3_runtime,
                sam3_model_id=sam3_model_id,
                sam3_load_error=sam3_load_error,
                sam3_threshold=sam3_threshold,
                sam3_mask_threshold=sam3_mask_threshold,
            )
            overlay = render_visualization(image, inference["bboxes"], inference["masks_rle"])
            stem = f"{timestamp}-{make_slug(query)}"
            save_paths = save_app_sample(
                output_dir=output_dir_path,
                stem=stem,
                image=image,
                inference=inference,
                query=query.strip(),
                task=task,
                source_label=source_label,
                save_artifacts=save_artifacts,
            )
            sample_record = {
                "index": 1,
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detections_count": len(inference["detections"]),
                "num_masks": inference["num_masks"],
                "generation_seconds": inference["generation_seconds"],
                "decoded_output": inference["decoded_output"],
                "detection_rows": flatten_detections(inference["detections"]),
                "save_paths": save_paths,
                "primary_engine": inference.get("primary_engine", "falcon"),
                "engines": inference.get("engines", []),
            }
            run_record["samples"].append(sample_record)
            run_record["current_image"] = image.copy()
            run_record["current_overlay"] = overlay.copy()
            run_record["primary_engine"] = inference.get("primary_engine", "falcon")
            run_record["engines"] = inference.get("engines", [])
            progress_box.progress(1.0, text="Image analysis complete.")
            render_banner(
                status_box,
                "success",
                "Image analysis complete",
                (
                    f"Detected {sample_record['detections_count']} regions with "
                    f"{title_case_engine(run_record['primary_engine'])} leading the pipeline."
                ),
            )
            with dashboard_box.container():
                render_run_dashboard(run_record, st.session_state["run_archive"])
        else:
            stream_url, fallback_label = resolve_stream_input(stream_preset, custom_stream_url)
            stream_info = resolve_stream_source(stream_url)
            run_record["stream_info"] = stream_info
            run_record["source_display"] = stream_info.get("title") or fallback_label

            render_banner(
                status_box,
                "info",
                "Connecting to livestream",
                f"Resolving and opening {run_record['source_display']} for {sample_count} sampled frames.",
            )
            progress_box.progress(0.05, text="Resolving livestream source...")
            capture = open_video_capture(stream_info, effective_settings["open_timeout"])

            try:
                for index in range(sample_count):
                    progress_box.progress(
                        index / sample_count,
                        text=f"Capturing frame {index + 1} of {sample_count}...",
                    )
                    image = read_frame_as_pil(capture, effective_settings["read_timeout"]).convert("RGB")
                    inference = run_orchestrated_inference_on_image(
                        engine=engine,
                        tokenizer=tokenizer,
                        model_args=model_args,
                        image=image,
                        query=query.strip(),
                        task=task,
                        min_dim=effective_settings["min_dim"],
                        max_dim=effective_settings["max_dim"],
                        max_new_tokens=effective_settings["max_new_tokens"],
                        temperature=0.0,
                        falcon_model_id=PERCEPTION_MODEL_ID,
                        enable_rtdetr=enable_rtdetr,
                        rtdetr_runtime=rtdetr_runtime,
                        rtdetr_model_id=rtdetr_model_id,
                        rtdetr_load_error=rtdetr_load_error,
                        rtdetr_threshold=rtdetr_threshold,
                        enable_sam3=enable_sam3,
                        sam3_runtime=sam3_runtime,
                        sam3_model_id=sam3_model_id,
                        sam3_load_error=sam3_load_error,
                        sam3_threshold=sam3_threshold,
                        sam3_mask_threshold=sam3_mask_threshold,
                    )
                    overlay = render_visualization(image, inference["bboxes"], inference["masks_rle"])
                    stem = f"{timestamp}-{make_slug(query)}-sample-{index + 1:02d}"
                    save_paths = save_app_sample(
                        output_dir=output_dir_path,
                        stem=stem,
                        image=image,
                        inference=inference,
                        query=query.strip(),
                        task=task,
                        source_label=stream_url,
                        save_artifacts=save_artifacts,
                    )

                    sample_record = {
                        "index": index + 1,
                        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "detections_count": len(inference["detections"]),
                        "num_masks": inference["num_masks"],
                        "generation_seconds": inference["generation_seconds"],
                        "decoded_output": inference["decoded_output"],
                        "detection_rows": flatten_detections(inference["detections"]),
                        "save_paths": save_paths,
                        "primary_engine": inference.get("primary_engine", "falcon"),
                        "engines": inference.get("engines", []),
                    }
                    run_record["samples"].append(sample_record)
                    run_record["current_image"] = image.copy()
                    run_record["current_overlay"] = overlay.copy()
                    run_record["primary_engine"] = inference.get("primary_engine", "falcon")
                    run_record["engines"] = inference.get("engines", [])

                    render_banner(
                        status_box,
                        "success",
                        f"Sample {index + 1} complete",
                        (
                            f"{run_record['source_display']} returned {sample_record['detections_count']} detections "
                            f"and {sample_record['num_masks']} masks with "
                            f"{title_case_engine(run_record['primary_engine'])} as the primary stage."
                        ),
                    )
                    progress_box.progress(
                        (index + 1) / sample_count,
                        text=(
                            f"Rendered sample {index + 1} of {sample_count} "
                            f"({format_seconds(sample_record['generation_seconds'])})"
                        ),
                    )
                    with dashboard_box.container():
                        render_run_dashboard(run_record, st.session_state["run_archive"])

                    if index + 1 < sample_count:
                        time.sleep(max(sample_interval, 0.0))
            finally:
                capture.release()

            progress_box.progress(1.0, text="Livestream monitoring run complete.")
    except Exception as exc:
        render_banner(status_box, "error", "Run failed", str(exc))
        progress_box.empty()
        if run_record["samples"]:
            with dashboard_box.container():
                render_run_dashboard(run_record, st.session_state["run_archive"])
        else:
            with dashboard_box.container():
                render_empty_state()
        return

    st.session_state["latest_run"] = run_record
    st.session_state["run_archive"] = [archive_entry_for_run(run_record), *st.session_state["run_archive"][:7]]


if __name__ == "__main__":
    main()
