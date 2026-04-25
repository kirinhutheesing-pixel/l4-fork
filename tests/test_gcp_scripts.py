import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GcpScriptContractTests(unittest.TestCase):
    def read_script(self, relative_path: str) -> str:
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_run_realtime_service_runs_source_then_sam_preflight_before_launch(self) -> None:
        script = self.read_script("scripts/gcp/run_realtime_service.sh")

        self.assertIn("--preflight-only", script)
        self.assertIn("--sam-preflight-only", script)
        self.assertIn('--sam3-model-id "${SAM3_MODEL_ID}"', script)
        self.assertIn('--display-max-fps "${DISPLAY_MAX_FPS}"', script)
        self.assertIn('--sam3-refresh-seconds "${SAM3_REFRESH_SECONDS}"', script)
        self.assertIn('if [[ -n "${YTDLP_COOKIES_FILE:-}" ]]', script)
        self.assertIn("--yt-cookies-file", script)

        source_preflight_index = script.index("preflight_cmd=(")
        sam_preflight_index = script.index("sam_preflight_cmd=(")
        container_remove_index = script.index('sudo docker rm -f "${CONTAINER_NAME}"')
        launch_index = script.index("launch_cmd=(")

        self.assertLess(source_preflight_index, sam_preflight_index)
        self.assertLess(sam_preflight_index, container_remove_index)
        self.assertLess(container_remove_index, launch_index)

    def test_check_realtime_service_requires_visible_sam3_primary(self) -> None:
        script = self.read_script("scripts/gcp/check_realtime_service.sh")

        self.assertIn('"primary_engine": result.get("primary_engine")', script)
        self.assertIn('"sam3_num_masks": sam3_num_masks', script)
        self.assertIn('readiness.get("sam3_visual_ready") is not True', script)
        self.assertIn('result.get("primary_engine") != "sam3"', script)
        self.assertIn('sam3_engine.get("status") != "ok"', script)
        self.assertIn("SAM3 did not report any masks", script)
        self.assertIn("CHECK_RESTAURANT_CONTRACT", script)
        self.assertIn("export CHECK_RESTAURANT_CONTRACT MAX_SAM3_AGE_SECONDS", script)
        self.assertIn("role_reason", script)
        self.assertIn("classification_source", script)

    def test_build_l4_image_wrapper_uses_detached_iap_build_and_log_polling(self) -> None:
        script = self.read_script("scripts/gcp/build_l4_image.ps1")

        self.assertIn("/tmp/falcon-pipeline-build.log", script)
        self.assertIn("--tunnel-through-iap", script)
        self.assertIn("git pull --ff-only", script)
        self.assertIn("nohup sudo docker build", script)
        self.assertIn("sudo docker image inspect", script)
        self.assertIn("BUILD_RUNNING", script)
        self.assertIn("IMAGE_READY", script)


if __name__ == "__main__":
    unittest.main()
