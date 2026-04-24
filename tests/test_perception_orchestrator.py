import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from perception_orchestrator import ModelAccessError, ModelUnsupportedError, load_sam3_runtime, run_sam3_inference


class FakeSam3Model:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return cls()

    def to(self, _device: str):
        return self

    def eval(self) -> None:
        return None


class FakeSam3Processor:
    calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.calls.append({"model_id": model_id, "kwargs": kwargs})
        return cls()


class GatedRepoError(RuntimeError):
    pass


class FailingSam3Processor:
    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs):
        raise GatedRepoError("401 Unauthorized. Access to model facebook/sam3 is restricted.")


class UnsupportedSam3Processor:
    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs):
        raise RuntimeError("Unrecognized model in checkpoint facebook/sam3.1.")


class AmbiguousSequence(list):
    def __bool__(self):
        raise RuntimeError("Boolean value is ambiguous")


class FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class FakeTorch:
    def inference_mode(self):
        return FakeInferenceMode()


class FakeRuntimeSam3Model:
    def __call__(self, **_inputs):
        return object()


class FakeRuntimeSam3Processor:
    def __call__(self, **_kwargs):
        return {"original_sizes": [[16, 16]]}

    def post_process_instance_segmentation(self, *_args, **_kwargs):
        return [
            {
                "boxes": AmbiguousSequence([[1.0, 2.0, 10.0, 12.0]]),
                "scores": AmbiguousSequence([0.87]),
                "masks": AmbiguousSequence([np.ones((16, 16), dtype=np.uint8)]),
            }
        ]


class Sam3RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeSam3Model.calls = []
        FakeSam3Processor.calls = []

    def test_load_sam3_runtime_passes_hf_token_to_transformers(self) -> None:
        transformers = types.SimpleNamespace(
            Sam3Model=FakeSam3Model,
            Sam3Processor=FakeSam3Processor,
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=False),
            mock.patch("perception_orchestrator._torch_device", return_value=(object(), "cuda")),
        ):
            runtime = load_sam3_runtime("facebook/sam3")

        self.assertEqual(runtime["model_id"], "facebook/sam3")
        self.assertEqual(FakeSam3Processor.calls[0]["kwargs"], {"token": "hf_test_token"})
        self.assertEqual(FakeSam3Model.calls[0]["kwargs"], {"token": "hf_test_token"})

    def test_load_sam3_runtime_surfaces_gated_repo_as_model_access(self) -> None:
        transformers = types.SimpleNamespace(
            Sam3Model=FakeSam3Model,
            Sam3Processor=FailingSam3Processor,
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch("perception_orchestrator._torch_device", return_value=(object(), "cuda")),
        ):
            with self.assertRaises(ModelAccessError) as caught:
                load_sam3_runtime("facebook/sam3")

        message = str(caught.exception)
        self.assertIn("model_access", message)
        self.assertIn("facebook/sam3", message)
        self.assertIn("HF_TOKEN", message)

    def test_load_sam3_runtime_surfaces_unsupported_checkpoint(self) -> None:
        transformers = types.SimpleNamespace(
            Sam3Model=FakeSam3Model,
            Sam3Processor=UnsupportedSam3Processor,
        )
        with (
            mock.patch.dict(sys.modules, {"transformers": transformers}),
            mock.patch("perception_orchestrator._torch_device", return_value=(object(), "cuda")),
        ):
            with self.assertRaises(ModelUnsupportedError) as caught:
                load_sam3_runtime("facebook/sam3.1")

        message = str(caught.exception)
        self.assertIn("model_unsupported", message)
        self.assertIn("facebook/sam3.1", message)

    def test_load_sam3_runtime_keeps_non_cuda_guard(self) -> None:
        with mock.patch("perception_orchestrator._torch_device", return_value=(object(), "cpu")):
            with self.assertRaises(RuntimeError) as caught:
                load_sam3_runtime("facebook/sam3")

        self.assertIn("disabled on non-CUDA", str(caught.exception))

    def test_run_sam3_inference_handles_tensor_like_result_sequences(self) -> None:
        runtime = {
            "torch": FakeTorch(),
            "processor": FakeRuntimeSam3Processor(),
            "model": FakeRuntimeSam3Model(),
            "device": "cuda",
            "model_id": "facebook/sam3",
        }

        with mock.patch("perception_orchestrator._encode_binary_mask", return_value={"counts": "x", "size": [16, 16]}):
            result = run_sam3_inference(
                runtime=runtime,
                image=Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)),
                query="person",
                prompt_bboxes=[{"x": 0.5, "y": 0.5, "w": 0.4, "h": 0.4}],
            )

        self.assertEqual(result["engine"], "sam3")
        self.assertEqual(len(result["detections"]), 1)
        self.assertEqual(result["num_masks"], 1)


if __name__ == "__main__":
    unittest.main()
