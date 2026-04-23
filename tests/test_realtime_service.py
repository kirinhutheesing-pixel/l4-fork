import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

from falcon_perception import PERCEPTION_300M_MODEL_ID
from falcon_pipeline_realtime_service import (
    ERROR_KIND_SOURCE_AUTH,
    ERROR_KIND_SOURCE_UNAVAILABLE,
    FalconPipelineRealtimeService,
    LiveSessionConfig,
    RuntimeOptions,
    SOURCE_STATUS_AUTH_REQUIRED,
    SOURCE_STATUS_IDLE,
    SOURCE_STATUS_READY,
    SOURCE_STATUS_UNAVAILABLE,
    build_restaurant_scene_annotations,
    create_app,
    parse_args,
    run_source_preflight,
)


class RealtimeServiceTests(unittest.TestCase):
    def make_service(
        self,
        *,
        source_url: str = "",
        load_rtdetr: bool = True,
        load_sam3: bool = False,
    ) -> FalconPipelineRealtimeService:
        temp_root = Path.cwd() / ".tmp-tests"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(dir=temp_root))
        self.addCleanup(shutil.rmtree, root, True)
        options = RuntimeOptions(
            cache_dir=root / "cache",
            output_dir=root / "outputs",
            load_rtdetr=load_rtdetr,
            load_sam3=load_sam3,
        )
        session = LiveSessionConfig(source_url=source_url)
        return FalconPipelineRealtimeService(options, session)

    def mark_models_loaded(self, service: FalconPipelineRealtimeService) -> None:
        service.falcon_runtime = object()
        service._set_model_state("falcon", "loaded")
        if service.options.load_rtdetr:
            service.rtdetr_runtime = {"model": object()}
            service._set_model_state("rt_detr", "loaded")
        if service.options.load_sam3:
            service.sam3_runtime = {"model": object()}
            service._set_model_state("sam3", "loaded")

    def test_parse_args_defaults_use_l4_profile(self) -> None:
        with mock.patch.object(sys, "argv", ["falcon_pipeline_realtime_service.py"]):
            args = parse_args()

        self.assertEqual(args.falcon_model_id, PERCEPTION_300M_MODEL_ID)
        self.assertEqual(args.task, "detection")
        self.assertFalse(args.enable_sam3)
        self.assertFalse(args.load_sam3)
        self.assertFalse(args.no_load_rtdetr)
        self.assertEqual(args.max_dim, 960)
        self.assertEqual(args.falcon_refresh_seconds, 2.0)

    def test_load_models_without_source_reports_idle_ready_state(self) -> None:
        service = self.make_service(source_url="", load_rtdetr=True, load_sam3=False)

        with (
            mock.patch("falcon_pipeline_realtime_service.FalconRealtimeRuntime", return_value=object()),
            mock.patch("falcon_pipeline_realtime_service.load_rtdetr_runtime", return_value={"model": object()}),
        ):
            service._load_models()

        state = service.get_state()
        self.assertEqual(state["readiness"]["service_state"], "idle")
        self.assertTrue(state["readiness"]["models_ready"])
        self.assertFalse(state["readiness"]["integration_ready"])
        self.assertTrue(state["frame"]["is_placeholder"])
        self.assertEqual(state["frame"]["note"], "Waiting for a stream source.")
        self.assertEqual(state["source"]["status"], SOURCE_STATUS_IDLE)
        self.assertFalse(state["source"]["cookie_file_configured"])
        self.assertIsNone(state["readiness"]["error_kind"])
        self.assertEqual(state["model_status"]["falcon"]["state"], "loaded")
        self.assertEqual(state["model_status"]["rt_detr"]["state"], "loaded")
        self.assertEqual(state["model_status"]["sam3"]["state"], "disabled")

    def test_load_models_with_source_reports_waiting_for_frame(self) -> None:
        service = self.make_service(
            source_url="https://www.youtube.com/watch?v=example",
            load_rtdetr=True,
            load_sam3=True,
        )

        with (
            mock.patch("falcon_pipeline_realtime_service.FalconRealtimeRuntime", return_value=object()),
            mock.patch("falcon_pipeline_realtime_service.load_rtdetr_runtime", return_value={"model": object()}),
            mock.patch("falcon_pipeline_realtime_service.load_sam3_runtime", return_value={"model": object()}),
        ):
            service._load_models()

        state = service.get_state()
        self.assertEqual(state["readiness"]["service_state"], "waiting_for_frame")
        self.assertTrue(state["readiness"]["models_ready"])
        self.assertFalse(state["frame"]["ready"])
        self.assertEqual(state["frame"]["note"], "Models loaded. Waiting for first live frame.")
        self.assertEqual(state["source"]["status"], "resolving")
        self.assertEqual(state["model_status"]["sam3"]["state"], "loaded")

    def test_load_model_errors_are_exposed(self) -> None:
        service = self.make_service(source_url="", load_rtdetr=True, load_sam3=True)

        with (
            mock.patch("falcon_pipeline_realtime_service.FalconRealtimeRuntime", return_value=object()),
            mock.patch(
                "falcon_pipeline_realtime_service.load_rtdetr_runtime",
                side_effect=RuntimeError("rtdetr failed"),
            ),
            mock.patch(
                "falcon_pipeline_realtime_service.load_sam3_runtime",
                side_effect=RuntimeError("sam3 failed"),
            ),
        ):
            service._load_models()

        state = service.get_state()
        self.assertEqual(state["model_status"]["falcon"]["state"], "loaded")
        self.assertEqual(state["model_status"]["rt_detr"]["state"], "error")
        self.assertEqual(state["model_status"]["sam3"]["state"], "error")
        self.assertEqual(state["rtdetr_load_error"], "rtdetr failed")
        self.assertEqual(state["sam3_load_error"], "sam3 failed")
        self.assertFalse(state["readiness"]["models_ready"])
        self.assertEqual(state["readiness"]["service_state"], "warming")

    def test_live_overlay_marks_service_ready(self) -> None:
        service = self.make_service(
            source_url="https://www.youtube.com/watch?v=example",
            load_rtdetr=True,
            load_sam3=False,
        )
        self.mark_models_loaded(service)
        service._set_metric("capture_state", "running")
        service._set_metric("pipeline_state", "running")
        overlay = np.zeros((32, 32, 3), dtype=np.uint8)
        service._set_latest_overlay(
            overlay,
            {
                "generation_seconds": 0.25,
                "detections": [],
                "bboxes": [],
                "masks_rle": [],
                "num_masks": 0,
            },
        )

        state = service.get_state()
        self.assertEqual(state["readiness"]["service_state"], "live")
        self.assertTrue(state["readiness"]["integration_ready"])
        self.assertEqual(state["frame"]["state"], "live")
        self.assertTrue(state["frame"]["ready"])
        self.assertFalse(state["frame"]["is_placeholder"])

    def test_session_endpoint_accepts_json_body(self) -> None:
        service = self.make_service(source_url="")
        app = create_app(service, "<html></html>")
        client = TestClient(app)

        response = client.post(
            "/api/session",
            json={
                "source_url": "https://www.youtube.com/watch?v=example",
                "prompt": "restaurant guests looking for service",
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["session"]["source_url"], "https://www.youtube.com/watch?v=example")
        self.assertEqual(payload["session"]["prompt"], "restaurant guests looking for service")

    def test_restaurant_scene_annotations_mark_service_seekers(self) -> None:
        inference = {
            "engine_outputs": {
                "rt_detr": {
                    "candidate_detections": [
                        {
                            "label": "person",
                            "center": {"x": 0.30, "y": 0.52},
                            "width": 0.10,
                            "height": 0.26,
                        },
                        {
                            "label": "person",
                            "center": {"x": 0.82, "y": 0.44},
                            "width": 0.09,
                            "height": 0.24,
                        },
                        {
                            "label": "dining table",
                            "center": {"x": 0.28, "y": 0.70},
                            "width": 0.30,
                            "height": 0.16,
                        },
                    ]
                },
                "falcon": {
                    "detections": [
                        {
                            "label": "restaurant guest looking for service",
                            "center": {"x": 0.31, "y": 0.53},
                            "width": 0.11,
                            "height": 0.26,
                        }
                    ]
                },
            }
        }

        scene = build_restaurant_scene_annotations(
            inference,
            "track restaurant goers, servers, and tables. person turns red if looking for service.",
        )

        self.assertIsNotNone(scene)
        assert scene is not None
        self.assertEqual(scene["profile"], "restaurant_service")
        self.assertEqual(scene["counts"]["tables"], 1)
        self.assertEqual(scene["counts"]["restaurant_goers"], 1)
        self.assertEqual(scene["counts"]["servers"], 1)
        self.assertEqual(scene["counts"]["needs_service"], 1)
        people = [entity for entity in scene["entities"] if entity["kind"] == "person"]
        self.assertEqual(len(people), 2)
        self.assertTrue(any(entity["needs_service"] for entity in people))
        self.assertTrue(any(entity["role"] == "server" for entity in people))
        self.assertTrue(all("bbox" in entity for entity in scene["entities"]))
        self.assertTrue(all("detection" not in entity for entity in scene["entities"]))

    def test_source_preflight_ready_source_returns_zero(self) -> None:
        capture = mock.Mock()
        capture.release = mock.Mock()
        with (
            mock.patch(
                "falcon_pipeline_realtime_service.resolve_stream_source",
                return_value={
                    "requested_url": "https://example.com/live",
                    "resolved_url": "https://example.com/stream.m3u8",
                    "source_type": "direct",
                    "title": "Example Live",
                    "channel": "Example",
                    "is_live": True,
                },
            ),
            mock.patch(
                "falcon_pipeline_realtime_service.open_video_capture",
                return_value=capture,
            ),
        ):
            payload, exit_code = run_source_preflight(
                source_url="https://example.com/live",
                cookie_file=Path("C:/tmp/youtube-cookies.txt"),
                timeout_seconds=12.0,
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["source"]["status"], SOURCE_STATUS_READY)
        self.assertTrue(payload["source"]["resolved_url_present"])
        self.assertTrue(payload["source"]["cookie_file_configured"])
        capture.release.assert_called_once()

    def test_source_preflight_passes_cookie_file_to_resolver(self) -> None:
        capture = mock.Mock()
        capture.release = mock.Mock()
        cookie_file = Path("C:/tmp/youtube-cookies.txt")
        with (
            mock.patch(
                "falcon_pipeline_realtime_service.resolve_stream_source",
                return_value={
                    "requested_url": "https://www.youtube.com/watch?v=example",
                    "resolved_url": "https://example.com/stream.m3u8",
                    "source_type": "youtube",
                    "title": "Example Live",
                    "channel": "Example",
                    "is_live": True,
                },
            ) as mock_resolve,
            mock.patch(
                "falcon_pipeline_realtime_service.open_video_capture",
                return_value=capture,
            ),
        ):
            payload, exit_code = run_source_preflight(
                source_url="https://www.youtube.com/watch?v=example",
                cookie_file=cookie_file,
                timeout_seconds=12.0,
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["ok"])
        mock_resolve.assert_called_once_with(
            "https://www.youtube.com/watch?v=example",
            cookie_file=cookie_file,
        )

    def test_source_preflight_auth_required_returns_20(self) -> None:
        with mock.patch(
            "falcon_pipeline_realtime_service.resolve_stream_source",
            side_effect=RuntimeError("Sign in to confirm you’re not a bot. Use --cookies-from-browser or --cookies for the authentication."),
        ):
            payload, exit_code = run_source_preflight(
                source_url="https://www.youtube.com/watch?v=example",
                cookie_file=None,
                timeout_seconds=12.0,
            )

        self.assertEqual(exit_code, 20)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["source"]["status"], SOURCE_STATUS_AUTH_REQUIRED)
        self.assertEqual(payload["source"]["error_kind"], ERROR_KIND_SOURCE_AUTH)

    def test_source_preflight_unavailable_returns_21(self) -> None:
        with mock.patch(
            "falcon_pipeline_realtime_service.resolve_stream_source",
            side_effect=RuntimeError("Could not resolve a playable media URL from the YouTube livestream."),
        ):
            payload, exit_code = run_source_preflight(
                source_url="https://www.youtube.com/watch?v=example",
                cookie_file=None,
                timeout_seconds=12.0,
            )

        self.assertEqual(exit_code, 21)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["source"]["status"], SOURCE_STATUS_UNAVAILABLE)
        self.assertEqual(payload["source"]["error_kind"], ERROR_KIND_SOURCE_UNAVAILABLE)

    def test_get_state_surfaces_source_error_and_scene_annotations(self) -> None:
        service = self.make_service(source_url="https://www.youtube.com/watch?v=example")
        self.mark_models_loaded(service)
        service._set_metric("capture_state", "error")
        service._set_metric("pipeline_state", "waiting_for_source")
        service._set_source_state(
            {
                "input_url": "https://www.youtube.com/watch?v=example",
                "source_type": "youtube",
                "status": SOURCE_STATUS_AUTH_REQUIRED,
                "resolved_url_present": False,
                "title": None,
                "channel": None,
                "is_live": None,
                "cookie_file_configured": False,
                "error_kind": ERROR_KIND_SOURCE_AUTH,
                "error_message": "Sign in required.",
            }
        )
        service._set_latest_overlay(
            np.zeros((24, 24, 3), dtype=np.uint8),
            {
                "detections": [],
                "bboxes": [],
                "masks_rle": [],
                "num_masks": 0,
                "scene_annotations": {
                    "profile": "restaurant_service",
                    "counts": {"tables": 1, "restaurant_goers": 2, "servers": 1, "needs_service": 1},
                    "entities": [
                        {
                            "entity_id": "person-1",
                            "kind": "person",
                            "role": "restaurant_goer",
                            "needs_service": True,
                            "bbox": {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.3},
                            "near_table": True,
                        }
                    ],
                },
            },
        )

        state = service.get_state()
        self.assertEqual(state["source"]["status"], SOURCE_STATUS_AUTH_REQUIRED)
        self.assertEqual(state["source"]["error_kind"], ERROR_KIND_SOURCE_AUTH)
        self.assertEqual(state["readiness"]["error_kind"], ERROR_KIND_SOURCE_AUTH)
        self.assertEqual(state["result"]["scene_annotations"]["profile"], "restaurant_service")
        self.assertEqual(state["result"]["scene_annotations"]["counts"]["needs_service"], 1)
        entity = state["result"]["scene_annotations"]["entities"][0]
        self.assertEqual(entity["kind"], "person")
        self.assertEqual(entity["role"], "restaurant_goer")
        self.assertTrue(entity["needs_service"])
        self.assertIn("bbox", entity)

    def test_source_preflight_missing_source_returns_21(self) -> None:
        payload, exit_code = run_source_preflight(
            source_url="",
            cookie_file=None,
            timeout_seconds=12.0,
        )

        self.assertEqual(exit_code, 21)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["source"]["status"], SOURCE_STATUS_UNAVAILABLE)
        self.assertEqual(payload["source"]["error_kind"], ERROR_KIND_SOURCE_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
