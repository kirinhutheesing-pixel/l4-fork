import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from falcon_perception import PERCEPTION_300M_MODEL_ID
from falcon_pipeline_realtime_service import (
    FalconPipelineRealtimeService,
    LiveSessionConfig,
    RuntimeOptions,
    parse_args,
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


if __name__ == "__main__":
    unittest.main()
