import io
import json
import sys
import types
import unittest
from pathlib import Path

import httpx
from PIL import Image


SERVICE_DIR = Path(__file__).resolve().parents[1] / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm"
sys.path.insert(0, str(SERVICE_DIR))


def _install_service_import_stubs():
    deepseek_mod = types.ModuleType("deepseek_ocr2")
    deepseek_mod.DeepseekOCR2ForCausalLM = type("DeepseekOCR2ForCausalLM", (), {})
    sys.modules["deepseek_ocr2"] = deepseek_mod

    process_mod = types.ModuleType("process")
    image_process_mod = types.ModuleType("process.image_process")
    image_process_mod.DeepseekOCR2Processor = type("DeepseekOCR2Processor", (), {})
    ngram_mod = types.ModuleType("process.ngram_norepeat")
    ngram_mod.NoRepeatNGramLogitsProcessor = type(
        "NoRepeatNGramLogitsProcessor",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    sys.modules["process"] = process_mod
    sys.modules["process.image_process"] = image_process_mod
    sys.modules["process.ngram_norepeat"] = ngram_mod

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.LLM = type("LLM", (), {})
    vllm_mod.SamplingParams = type(
        "SamplingParams",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    registry_mod = types.ModuleType("vllm.model_executor.models.registry")
    registry_mod.ModelRegistry = type(
        "ModelRegistry",
        (),
        {"register_model": staticmethod(lambda *args, **kwargs: None)},
    )
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.model_executor"] = types.ModuleType("vllm.model_executor")
    sys.modules["vllm.model_executor.models"] = types.ModuleType("vllm.model_executor.models")
    sys.modules["vllm.model_executor.models.registry"] = registry_mod


_install_service_import_stubs()

import start_service  # noqa: E402


class _FakeCompletion:
    def __init__(self, text: str):
        self.text = text
        self.token_ids = [1, 2]


class _FakeOutput:
    def __init__(self, text: str):
        self.prompt_token_ids = [1, 2, 3]
        self.outputs = [_FakeCompletion(text)]


class _FakeProcessor:
    def tokenize_with_images(self, images, bos, eos, cropping, prompt):
        return images[0].info.get("label", "loaded")


class _FakeLLM:
    def generate(self, batch_inputs, sampling_params):
        return [
            _FakeOutput(f"raw:{item['multi_modal_data']['image']}")
            for item in batch_inputs
        ]


def _image(label: str) -> Image.Image:
    image = Image.new("RGB", (16, 16), "white")
    image.info["label"] = label
    return image


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buf, format="PNG")
    return buf.getvalue()


class DeepSeekServiceResponseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = {
            "_llm": start_service._llm,
            "_sampling_params": start_service._sampling_params,
            "_processor": start_service._processor,
            "_microbatcher": start_service._microbatcher,
            "_microbatch_enabled": start_service._microbatch_enabled,
            "_pdf_to_images": start_service._pdf_to_images,
            "_num_workers": start_service._num_workers,
            "_render_workers": start_service._render_workers,
            "_render_executor": start_service._render_executor,
            "_preprocess_executor": start_service._preprocess_executor,
        }
        start_service._llm = _FakeLLM()
        start_service._sampling_params = object()
        start_service._processor = _FakeProcessor()
        start_service._microbatch_enabled = True
        start_service._num_workers = 1
        start_service._microbatcher = start_service.MicroBatcher(
            max_wait_ms=1,
            max_pages=64,
            queue_pages=128,
            generate_in_thread=False,
        )
        start_service._pdf_to_images = lambda pdf_bytes, dpi=144: [
            _image("pdf-0"),
            _image("pdf-1"),
        ]
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=start_service.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        if start_service._microbatcher is not None:
            await start_service._microbatcher.close()
        start_service._shutdown_cpu_executors()
        for name, value in self.originals.items():
            setattr(start_service, name, value)

    def assert_usage(self, body, *, pages: int):
        self.assertIn("usage", body)
        self.assertEqual(body["usage"]["input_tokens"], pages * 3)
        self.assertEqual(body["usage"]["cached_tokens"], 0)
        self.assertEqual(body["usage"]["output_tokens"], pages * 2)
        self.assertEqual(body["usage"]["total_tokens"], pages * 5)
        self.assertEqual(body["usage"]["prompt_tokens"], pages * 3)
        self.assertEqual(body["usage"]["completion_tokens"], pages * 2)
        self.assertEqual(body["usage"]["input_tokens_details"], {"cached_tokens": 0})
        for result in body["results"]:
            self.assertNotIn("_usage", result)
            self.assertNotIn("_batch_id", result)
            self.assertNotIn("_queue_wait_ms", result)

    def assert_timing(self, response, *, expected_step: str, pages: int):
        self.assertIn("x-kie-timing", response.headers)
        timing = json.loads(response.headers["x-kie-timing"])
        self.assertIsInstance(timing["total_ms"], int)
        self.assertEqual(timing["page_count"], pages)
        self.assertIn(expected_step, timing["steps"])
        self.assertIn("cpu_preprocess", timing["steps"])
        self.assertIn("gpu_generate", timing["steps"])
        self.assertIn("postprocess_cpu", timing["steps"])
        self.assertIn("response_build", timing["steps"])

    async def test_predict_pdf_response_includes_usage(self):
        response = await self.client.post("/predict/pdf", content=b"%PDF-1.7")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"page_count", "results", "usage"})
        self.assertEqual(body["page_count"], 2)
        self.assertEqual(len(body["results"]), 2)
        self.assertEqual(body["results"][0]["raw_text"], "raw:pdf-0")
        self.assertEqual(body["results"][1]["raw_text"], "raw:pdf-1")
        self.assert_usage(body, pages=2)
        self.assert_timing(response, expected_step="pdf_render_cpu", pages=2)

    async def test_predict_image_response_includes_usage(self):
        response = await self.client.post("/predict/image", content=_png_bytes())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"results", "usage"})
        self.assertEqual(len(body["results"]), 1)
        self.assertIn("raw_text", body["results"][0])
        self.assertIn("image_size", body["results"][0])
        self.assert_usage(body, pages=1)
        self.assert_timing(response, expected_step="image_decode_cpu", pages=1)

    async def test_predict_batch_response_includes_usage(self):
        import base64

        payload = {
            "images": [
                base64.b64encode(_png_bytes()).decode("ascii"),
                base64.b64encode(_png_bytes()).decode("ascii"),
            ]
        }
        response = await self.client.post("/predict/batch", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"results", "usage"})
        self.assertEqual(len(body["results"]), 2)
        self.assertIn("raw_text", body["results"][0])
        self.assert_usage(body, pages=2)
        self.assert_timing(response, expected_step="image_decode_cpu", pages=2)


if __name__ == "__main__":
    unittest.main()
