import asyncio
import sys
import types
import unittest
from pathlib import Path

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
        self.token_ids = list(range(max(1, len(text))))


class _FakeOutput:
    def __init__(self, text: str):
        self.prompt_token_ids = [1, 2, 3]
        self.outputs = [_FakeCompletion(text)]


class _FakeProcessor:
    def tokenize_with_images(self, images, bos, eos, cropping, prompt):
        return images[0].info.get("label", "unlabeled")


class _FakeLLM:
    def __init__(self, *, fail: bool = False, output_count: int | None = None):
        self.calls = []
        self.fail = fail
        self.output_count = output_count

    def generate(self, batch_inputs, sampling_params):
        self.calls.append(list(batch_inputs))
        if self.fail:
            raise RuntimeError("boom")
        outputs = [
            _FakeOutput(f"raw:{item['multi_modal_data']['image']}")
            for item in batch_inputs
        ]
        if self.output_count is not None:
            return outputs[:self.output_count]
        return outputs


def _image(label: str) -> Image.Image:
    image = Image.new("RGB", (16, 16), "white")
    image.info["label"] = label
    return image


class MicroBatchingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.originals = {
            "_llm": start_service._llm,
            "_sampling_params": start_service._sampling_params,
            "_processor": start_service._processor,
            "_microbatcher": start_service._microbatcher,
            "_microbatch_enabled": start_service._microbatch_enabled,
            "_num_workers": start_service._num_workers,
        }
        start_service._sampling_params = object()
        start_service._processor = _FakeProcessor()
        start_service._microbatch_enabled = True
        start_service._num_workers = 1

    async def asyncTearDown(self):
        if start_service._microbatcher is not None:
            await start_service._microbatcher.close()
        for name, value in self.originals.items():
            setattr(start_service, name, value)

    async def test_concurrent_requests_share_one_generate_call(self):
        fake_llm = _FakeLLM()
        start_service._llm = fake_llm
        batcher = start_service.MicroBatcher(
            max_wait_ms=100,
            max_pages=10,
            queue_pages=100,
            generate_in_thread=False,
        )
        start_service._microbatcher = batcher

        results = await asyncio.gather(
            batcher.submit_many([_image("a0")], "prompt", "req-a"),
            batcher.submit_many([_image("b0"), _image("b1")], "prompt", "req-b"),
            batcher.submit_many([_image("c0")], "prompt", "req-c"),
        )

        self.assertEqual(len(fake_llm.calls), 1)
        self.assertEqual(len(fake_llm.calls[0]), 4)
        self.assertEqual(results[0][0]["raw_text"], "raw:a0")
        self.assertEqual([r["raw_text"] for r in results[1]], ["raw:b0", "raw:b1"])
        self.assertEqual(results[2][0]["raw_text"], "raw:c0")

    async def test_max_pages_splits_large_queue(self):
        fake_llm = _FakeLLM()
        start_service._llm = fake_llm
        batcher = start_service.MicroBatcher(
            max_wait_ms=1,
            max_pages=2,
            queue_pages=100,
            generate_in_thread=False,
        )
        start_service._microbatcher = batcher

        results = await batcher.submit_many(
            [_image(f"p{i}") for i in range(5)],
            "prompt",
            "req",
        )

        self.assertEqual([len(call) for call in fake_llm.calls], [2, 2, 1])
        self.assertEqual([result["raw_text"] for result in results], [
            "raw:p0",
            "raw:p1",
            "raw:p2",
            "raw:p3",
            "raw:p4",
        ])

    async def test_generate_exception_propagates_to_request(self):
        start_service._llm = _FakeLLM(fail=True)
        batcher = start_service.MicroBatcher(
            max_wait_ms=1,
            max_pages=10,
            queue_pages=100,
            generate_in_thread=False,
        )
        start_service._microbatcher = batcher

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await batcher.submit_many([_image("x")], "prompt", "req")

    async def test_output_count_mismatch_propagates_to_request(self):
        start_service._llm = _FakeLLM(output_count=1)
        batcher = start_service.MicroBatcher(
            max_wait_ms=1,
            max_pages=10,
            queue_pages=100,
            generate_in_thread=False,
        )
        start_service._microbatcher = batcher

        with self.assertRaisesRegex(RuntimeError, "1 outputs for 2 inputs"):
            await batcher.submit_many([_image("x"), _image("y")], "prompt", "req")

    async def test_processor_is_initialized_once_before_parallel_preprocess(self):
        fake_llm = _FakeLLM()
        start_service._llm = fake_llm
        start_service._processor = None

        class CountingProcessor:
            constructed = 0

            def __init__(self):
                type(self).constructed += 1

            def tokenize_with_images(self, images, bos, eos, cropping, prompt):
                return images[0].info.get("label", "unlabeled")

        image_process_mod = sys.modules["process.image_process"]
        original_processor = image_process_mod.DeepseekOCR2Processor
        image_process_mod.DeepseekOCR2Processor = CountingProcessor
        try:
            results = start_service._run_inference_inputs(
                [
                    (_image("x"), "prompt"),
                    (_image("y"), "prompt"),
                    (_image("z"), "prompt"),
                ],
                parallel_preprocess=True,
            )
        finally:
            image_process_mod.DeepseekOCR2Processor = original_processor

        self.assertEqual(CountingProcessor.constructed, 1)
        self.assertEqual([result["raw_text"] for result in results], [
            "raw:x",
            "raw:y",
            "raw:z",
        ])

    async def test_cancelled_future_is_skipped_before_generation(self):
        fake_llm = _FakeLLM()
        start_service._llm = fake_llm
        batcher = start_service.MicroBatcher(
            max_wait_ms=1,
            max_pages=10,
            queue_pages=100,
            generate_in_thread=False,
        )
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.cancel()
        prepared = start_service._preprocess_prepared_input((0, _image("x"), "prompt"))
        item = start_service._BatchItem(
            request_id="req",
            page_index=0,
            batch_input=prepared.batch_input,
            image_size=prepared.image_size,
            future=future,
            queued_at=loop.time(),
        )

        await batcher._run_batch([item])

        self.assertEqual(fake_llm.calls, [])

    async def test_close_fails_request_waiting_in_batch_window(self):
        start_service._llm = _FakeLLM()
        batcher = start_service.MicroBatcher(
            max_wait_ms=10_000,
            max_pages=10,
            queue_pages=100,
            generate_in_thread=False,
        )
        start_service._microbatcher = batcher

        task = asyncio.create_task(batcher.submit_many([_image("x")], "prompt", "req"))
        await asyncio.sleep(0.01)
        await batcher.close()

        with self.assertRaisesRegex(RuntimeError, "Microbatcher closed"):
            await task

    async def test_submit_after_close_is_rejected(self):
        start_service._llm = _FakeLLM()
        batcher = start_service.MicroBatcher(
            max_wait_ms=1,
            max_pages=10,
            queue_pages=100,
            generate_in_thread=False,
        )
        start_service._microbatcher = batcher

        await batcher.close()

        with self.assertRaisesRegex(RuntimeError, "Microbatcher is closed"):
            await batcher.submit_many([_image("x")], "prompt", "req")

    async def test_empty_prepared_request_skips_generate(self):
        fake_llm = _FakeLLM()
        start_service._llm = fake_llm
        start_service._microbatch_enabled = False
        timer = start_service.StepTimer()

        results = await start_service._run_prepared_inference_for_request(
            [],
            "req",
            timer,
        )

        self.assertEqual(results, [])
        self.assertEqual(fake_llm.calls, [])
        self.assertEqual(timer.summary()["gpu_batches"], 0)


if __name__ == "__main__":
    unittest.main()
