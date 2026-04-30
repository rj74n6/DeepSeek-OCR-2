import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _install_service_import_stubs():
    deepseek_mod = types.ModuleType("deepseek_ocr2")
    deepseek_mod.DeepseekOCR2ForCausalLM = type("DeepseekOCR2ForCausalLM", (), {})
    sys.modules["deepseek_ocr2"] = deepseek_mod

    process_mod = types.ModuleType("process")
    image_process_mod = types.ModuleType("process.image_process")
    image_process_mod.DeepseekOCR2Processor = _FakeProcessor
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


class _FakeProcessor:
    def tokenize_with_images(self, images, bos, eos, cropping, prompt):
        return images[0].info["label"]


_install_service_import_stubs()

from tools import offline_ocr_batch  # noqa: E402


class _FakeCompletion:
    def __init__(self, text):
        self.text = text
        self.token_ids = [1, 2]


class _FakeOutput:
    def __init__(self, text):
        self.prompt_token_ids = [1, 2, 3]
        self.outputs = [_FakeCompletion(text)]


class _FakeLLM:
    def __init__(self, *, fail_calls=None):
        self.calls = []
        self.fail_calls = set(fail_calls or [])

    def generate(self, batch_inputs, sampling_params):
        self.calls.append(list(batch_inputs))
        if len(self.calls) in self.fail_calls:
            raise RuntimeError(f"fail call {len(self.calls)}")
        return [
            _FakeOutput(f"raw:{item['multi_modal_data']['image']}")
            for item in batch_inputs
        ]


def _image(label):
    image = Image.new("RGB", (16, 16), "white")
    image.info["label"] = label
    return image


class OfflineOcrBatchTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "_llm": offline_ocr_batch.service._llm,
            "_sampling_params": offline_ocr_batch.service._sampling_params,
            "_init_model": offline_ocr_batch.service._init_model,
            "discover_documents": offline_ocr_batch.discover_documents,
            "_render_document": offline_ocr_batch._render_document,
        }
        offline_ocr_batch.service._init_model = lambda **kwargs: None
        offline_ocr_batch.service._sampling_params = object()
        offline_ocr_batch.PROCESSOR = _FakeProcessor()

    def tearDown(self):
        for name, value in self.originals.items():
            if name in {"discover_documents", "_render_document"}:
                setattr(offline_ocr_batch, name, value)
            else:
                setattr(offline_ocr_batch.service, name, value)
        offline_ocr_batch.PROCESSOR = None

    def _config(self, tmpdir, *, batch_pages=2):
        root = Path(tmpdir)
        input_dir = root / "input"
        input_dir.mkdir()
        return offline_ocr_batch.RunnerConfig(
            input_folder=input_dir,
            model_path="model",
            output_jsonl=root / "out.jsonl",
            metrics_json=root / "metrics.json",
            batch_pages=batch_pages,
            preprocess_executor="serial",
            num_workers=1,
        )

    def test_writes_documents_in_order_and_aggregates_usage(self):
        fake_llm = _FakeLLM()
        offline_ocr_batch.service._llm = fake_llm
        docs = [Path("a.pdf"), Path("b.pdf")]
        images = {
            "a.pdf": [_image("a0"), _image("a1"), _image("a2")],
            "b.pdf": [_image("b0")],
        }
        offline_ocr_batch.discover_documents = lambda folder, limit_docs=None: docs
        offline_ocr_batch._render_document = lambda path, dpi: images[path.name]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, batch_pages=2)
            metrics = offline_ocr_batch.OfflineOcrRunner(config).run()
            lines = [
                json.loads(line)
                for line in config.output_jsonl.read_text(encoding="utf-8").splitlines()
            ]
            metrics_file = json.loads(config.metrics_json.read_text(encoding="utf-8"))

        self.assertEqual([line["document_path"] for line in lines], ["a.pdf", "b.pdf"])
        self.assertEqual(lines[0]["status"], "ok")
        self.assertEqual(lines[0]["page_count"], 3)
        self.assertEqual(
            [result["raw_text"] for result in lines[0]["results"]],
            ["raw:a0", "raw:a1", "raw:a2"],
        )
        self.assertEqual(lines[1]["results"][0]["raw_text"], "raw:b0")
        self.assertEqual(lines[0]["usage"]["input_tokens"], 9)
        self.assertEqual(lines[0]["usage"]["output_tokens"], 6)
        self.assertEqual([len(call) for call in fake_llm.calls], [2, 2])
        self.assertEqual(metrics["documents"], 2)
        self.assertEqual(metrics["succeeded"], 2)
        self.assertEqual(metrics["failed"], 0)
        self.assertEqual(metrics["pages"], 4)
        self.assertEqual(metrics["generated_pages"], 4)
        self.assertEqual(metrics_file["pages"], 4)
        self.assertEqual(metrics_file["generated_pages"], 4)
        self.assertEqual(metrics_file["total_tokens"], 20)
        self.assertEqual(metrics_file["generated_total_tokens"], 20)

    def test_batch_failure_marks_only_documents_in_failed_batch(self):
        fake_llm = _FakeLLM(fail_calls={1})
        offline_ocr_batch.service._llm = fake_llm
        docs = [Path("a.pdf"), Path("b.pdf"), Path("c.pdf")]
        images = {
            "a.pdf": [_image("a0")],
            "b.pdf": [_image("b0"), _image("b1")],
            "c.pdf": [_image("c0")],
        }
        offline_ocr_batch.discover_documents = lambda folder, limit_docs=None: docs
        offline_ocr_batch._render_document = lambda path, dpi: images[path.name]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, batch_pages=2)
            metrics = offline_ocr_batch.OfflineOcrRunner(config).run()
            lines = [
                json.loads(line)
                for line in config.output_jsonl.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([line["document_path"] for line in lines], ["a.pdf", "b.pdf", "c.pdf"])
        self.assertEqual([line["status"] for line in lines], ["failed", "failed", "ok"])
        self.assertIn("fail call 1", lines[0]["error"])
        self.assertIn("fail call 1", lines[1]["error"])
        self.assertEqual(lines[2]["results"][0]["raw_text"], "raw:c0")
        self.assertEqual(metrics["succeeded"], 1)
        self.assertEqual(metrics["failed"], 2)
        self.assertEqual(metrics["pages"], 1)
        self.assertEqual(metrics["generated_pages"], 1)
        self.assertEqual(len(metrics["failures"]), 2)

    def test_partial_failed_document_does_not_inflate_success_metrics(self):
        fake_llm = _FakeLLM(fail_calls={2})
        offline_ocr_batch.service._llm = fake_llm
        docs = [Path("a.pdf")]
        images = {"a.pdf": [_image("a0"), _image("a1")]}
        offline_ocr_batch.discover_documents = lambda folder, limit_docs=None: docs
        offline_ocr_batch._render_document = lambda path, dpi: images[path.name]

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir, batch_pages=1)
            metrics = offline_ocr_batch.OfflineOcrRunner(config).run()
            lines = [
                json.loads(line)
                for line in config.output_jsonl.read_text(encoding="utf-8").splitlines()
            ]
            metrics_file = json.loads(config.metrics_json.read_text(encoding="utf-8"))

        self.assertEqual(lines[0]["status"], "failed")
        self.assertEqual(lines[0]["results"], [])
        self.assertEqual(metrics["succeeded"], 0)
        self.assertEqual(metrics["failed"], 1)
        self.assertEqual(metrics["pages"], 0)
        self.assertEqual(metrics["total_tokens"], 0)
        self.assertEqual(metrics["generated_pages"], 1)
        self.assertEqual(metrics["generated_total_tokens"], 5)
        self.assertEqual(metrics_file["pages"], 0)
        self.assertEqual(metrics_file["generated_pages"], 1)


if __name__ == "__main__":
    unittest.main()
