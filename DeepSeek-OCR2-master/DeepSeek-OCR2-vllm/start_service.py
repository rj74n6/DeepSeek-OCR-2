"""DeepSeek-OCR-2 model service based on vLLM.

Usage (conda env: deepseek-ocr2):
    python start_service.py --model-path /path/to/model --port 8300
    python start_service.py --model-path /path/to/model --gpu-memory-utilization 0.85
"""

import argparse
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import re
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch

if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["VLLM_USE_V1"] = "0"

import fitz
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps
from uvicorn.config import LOGGING_CONFIG

LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
ACCESS_LOG_FORMAT = (
    '%(asctime)s %(levelname)s [%(name)s] %(client_addr)s - '
    '"%(request_line)s" %(status_code)s'
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_CONCURRENCY = 128
DEFAULT_GPU_MEM_UTIL = 0.3
DEFAULT_CROP_MODE = True
DEFAULT_NUM_WORKERS = 64
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_MICROBATCH_MAX_WAIT_MS = 150
DEFAULT_MICROBATCH_MAX_PAGES = 64
DEFAULT_MICROBATCH_QUEUE_PAGES = 512
DEFAULT_RENDER_WORKERS = min(8, os.cpu_count() or 4)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_llm: Optional[LLM] = None
_sampling_params: Optional[SamplingParams] = None
_crop_mode: bool = DEFAULT_CROP_MODE
_num_workers: int = DEFAULT_NUM_WORKERS
_processor: Optional[Any] = None
_processor_lock = threading.Lock()
_microbatcher: Optional["MicroBatcher"] = None
_microbatch_enabled: bool = True
_render_workers: int = DEFAULT_RENDER_WORKERS
_render_executor: Optional[ThreadPoolExecutor] = None
_preprocess_executor: Optional[ThreadPoolExecutor] = None


@contextlib.asynccontextmanager
async def _lifespan(_: FastAPI):
    try:
        yield
    finally:
        if _microbatcher is not None:
            await _microbatcher.close()
        _shutdown_cpu_executors()


app = FastAPI(title="DeepSeek-OCR-2 Service", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StepTimer:
    """Small KIE-style wall-clock timer for response timing headers."""

    _RESERVED_META_KEYS = frozenset({"total_ms", "steps"})

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._steps: Dict[str, float] = {}
        self._meta: Dict[str, Any] = {}

    @contextlib.contextmanager
    def step(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.record(name, (time.monotonic() - t0) * 1000)

    def record(self, name: str, ms: float) -> None:
        self._steps[name] = self._steps.get(name, 0.0) + max(0.0, ms)

    def record_meta(self, key: str, value: Any) -> None:
        if key in self._RESERVED_META_KEYS:
            raise ValueError(f"meta key {key!r} collides with a reserved timing field")
        self._meta[key] = value

    def summary(self) -> Dict[str, Any]:
        total_ms = (time.monotonic() - self._start) * 1000
        steps = dict(self._steps)
        accounted = sum(steps.values())
        unaccounted = total_ms - accounted
        if unaccounted >= 0.5:
            steps["unaccounted"] = unaccounted
        out: Dict[str, Any] = {
            "total_ms": round(total_ms),
            "steps": {k: round(v) for k, v in steps.items()},
        }
        out.update(self._meta)
        return out


_TIMING_STEP_DEFAULTS = (
    "request_body",
    "pdf_render_cpu",
    "image_decode_cpu",
    "cpu_preprocess",
    "gpu_queue_wait",
    "gpu_generate",
    "postprocess_cpu",
    "response_build",
)


def _timed_json_response(
    content: Any,
    timer: StepTimer,
    *,
    status_code: int = 200,
) -> JSONResponse:
    for name in _TIMING_STEP_DEFAULTS:
        timer._steps.setdefault(name, 0.0)
    with timer.step("response_build"):
        response = JSONResponse(content, status_code=status_code)
    response.headers["x-kie-timing"] = json.dumps(
        timer.summary(), separators=(",", ":")
    )
    return response


def _get_render_executor() -> ThreadPoolExecutor:
    global _render_executor  # noqa: PLW0603
    if _render_executor is None:
        _render_executor = ThreadPoolExecutor(
            max_workers=max(1, _render_workers),
            thread_name_prefix="deepseek-render",
        )
    return _render_executor


def _get_preprocess_executor() -> ThreadPoolExecutor:
    global _preprocess_executor  # noqa: PLW0603
    if _preprocess_executor is None:
        _preprocess_executor = ThreadPoolExecutor(
            max_workers=max(1, _num_workers),
            thread_name_prefix="deepseek-preprocess",
        )
    return _preprocess_executor


def _shutdown_cpu_executors() -> None:
    global _render_executor, _preprocess_executor  # noqa: PLW0603
    if _render_executor is not None:
        _render_executor.shutdown(wait=False, cancel_futures=True)
        _render_executor = None
    if _preprocess_executor is not None:
        _preprocess_executor.shutdown(wait=False, cancel_futures=True)
        _preprocess_executor = None

def _uvicorn_log_config() -> dict:
    config = deepcopy(LOGGING_CONFIG)
    config["formatters"]["default"]["fmt"] = LOG_FORMAT
    config["formatters"]["default"]["datefmt"] = LOG_DATE_FORMAT
    config["formatters"]["access"]["fmt"] = ACCESS_LOG_FORMAT
    config["formatters"]["access"]["datefmt"] = LOG_DATE_FORMAT
    return config


def _init_model(
    model_path: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    cuda_devices: Optional[str],
    enforce_eager: bool,
) -> None:
    global _llm, _sampling_params  # noqa: PLW0603

    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    from deepseek_ocr2 import DeepseekOCR2ForCausalLM

    ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

    LOGGER.info(
        "Loading DeepSeek-OCR-2 model from '%s' (max_model_len=%d, max_num_seqs=%d, gpu_mem=%.2f, tp=%d)",
        model_path, max_model_len, max_num_seqs, gpu_memory_utilization, tensor_parallel_size,
    )

    _llm = LLM(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        max_model_len=max_model_len,
        swap_space=0,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_mm_preprocessor_cache=True,
    )

    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20, window_size=50,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        )
    ]

    _sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_model_len,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )

    LOGGER.info("DeepSeek-OCR-2 model loaded successfully.")


def _pdf_to_images(pdf_bytes: bytes, dpi: int = 144) -> List[Image.Image]:
    """Convert PDF bytes to a list of PIL Images."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def _load_image_bytes(image_bytes: bytes) -> Image.Image:
    """Load an image from raw bytes, applying EXIF orientation correction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _get_processor() -> Any:
    global _processor  # noqa: PLW0603
    if _processor is None:
        with _processor_lock:
            if _processor is None:
                from process.image_process import DeepseekOCR2Processor

                _processor = DeepseekOCR2Processor()
    return _processor


def _preprocess_single(image: Image.Image) -> dict:
    """Tokenize a single image for the model."""
    prompt = DEFAULT_PROMPT
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": _get_processor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=_crop_mode, prompt=prompt,
            )
        },
    }


def _preprocess_single_with_prompt(args: tuple) -> dict:
    """Tokenize a single image with a custom prompt."""
    image, prompt = args
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": _get_processor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=_crop_mode, prompt=prompt,
            )
        },
    }


def _clean_output(text: str) -> str:
    """Clean model output: remove EOS token and ref/det tags."""
    if "<｜end▁of▁sentence｜>" in text:
        text = text.replace("<｜end▁of▁sentence｜>", "")

    # Remove detection annotations but keep the referenced text
    pattern = r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>"
    matches = re.findall(pattern, text, re.DOTALL)
    for full_match_groups in matches:
        label, coords = full_match_groups
        full_tag = f"<|ref|>{label}<|/ref|><|det|>{coords}<|/det|>"
        if label == "image":
            text = text.replace(full_tag, "")
        else:
            text = text.replace(full_tag, "")

    text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_detections(text: str, image_width: int, image_height: int) -> List[Dict[str, Any]]:
    """Extract bounding-box detections from model output."""
    pattern = r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>"
    matches = re.findall(pattern, text, re.DOTALL)

    detections = []
    for label, coords_str in matches:
        try:
            coords_list = eval(coords_str)  # noqa: S307
        except Exception:
            continue
        for coords in coords_list:
            x1, y1, x2, y2 = coords
            detections.append({
                "label": label,
                "bbox_norm": [x1, y1, x2, y2],
                "bbox": [
                    int(x1 / 999 * image_width),
                    int(y1 / 999 * image_height),
                    int(x2 / 999 * image_width),
                    int(y2 / 999 * image_height),
                ],
            })
    return detections


def _request_id() -> str:
    return uuid4().hex[:12]


def _token_count(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _output_token_count(output: Any) -> int:
    total = 0
    for item in getattr(output, "outputs", []) or []:
        total += _token_count(getattr(item, "token_ids", None))
    return total


def _usage_for_output(output: Any) -> Dict[str, int]:
    prompt_tokens = _token_count(getattr(output, "prompt_token_ids", None))
    output_tokens = _output_token_count(output)
    return {
        "input_tokens": prompt_tokens,
        "cached_tokens": 0,
        "output_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }


def _response_usage_from_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    input_tokens = 0
    cached_tokens = 0
    output_tokens = 0
    total_tokens = 0

    for result in results:
        usage = result.pop("_usage", None) or {}
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        cached_tokens += int(usage.get("cached_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)
        total_tokens += int(usage.get("total_tokens", 0) or 0)

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "input_tokens_details": {"cached_tokens": cached_tokens},
    }


def _outputs_to_results_for_sizes(
    outputs_list: list[Any],
    image_sizes: list[tuple[int, int]],
) -> List[Dict[str, Any]]:
    if len(outputs_list) != len(image_sizes):
        raise RuntimeError(
            f"Model returned {len(outputs_list)} outputs for {len(image_sizes)} inputs."
        )

    results = []
    for output, (w, h) in zip(outputs_list, image_sizes):
        raw_text = output.outputs[0].text
        has_eos = "<｜end▁of▁sentence｜>" in raw_text
        results.append({
            "raw_text": raw_text,
            "clean_text": _clean_output(raw_text),
            "detections": _extract_detections(raw_text, w, h),
            "has_eos": has_eos,
            "image_size": {"width": w, "height": h},
            "_usage": _usage_for_output(output),
        })
    return results


def _outputs_to_results(outputs_list: list[Any], images: list[Image.Image]) -> List[Dict[str, Any]]:
    return _outputs_to_results_for_sizes(outputs_list, [img.size for img in images])


@dataclass
class _PreparedInput:
    page_index: int
    batch_input: dict
    image_size: tuple[int, int]


def _preprocess_prepared_input(args: tuple) -> _PreparedInput:
    page_index, image, prompt = args
    image_size = image.size
    return _PreparedInput(
        page_index=page_index,
        batch_input=_preprocess_single_with_prompt((image, prompt)),
        image_size=image_size,
    )


async def _prepare_images_for_inference(
    images: List[Image.Image],
    prompt: str,
    timer: StepTimer,
) -> list[_PreparedInput]:
    """CPU-preprocess images into vLLM-ready batch inputs."""
    if not images:
        return []

    loop = asyncio.get_running_loop()
    prepared: list[_PreparedInput]
    try:
        with timer.step("cpu_preprocess"):
            _get_processor()
            futures = [
                loop.run_in_executor(
                    _get_preprocess_executor(),
                    _preprocess_prepared_input,
                    (page_index, image, prompt),
                )
                for page_index, image in enumerate(images)
            ]
            gathered = await asyncio.gather(*futures, return_exceptions=True)
            errors = [item for item in gathered if isinstance(item, BaseException)]
            if errors:
                raise errors[0]
            prepared = [item for item in gathered if isinstance(item, _PreparedInput)]
    finally:
        for image in images:
            with contextlib.suppress(Exception):
                image.close()

    prepared.sort(key=lambda item: item.page_index)
    return prepared


def _run_prepared_batch_sync(
    batch_inputs: list[dict],
    image_sizes: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if _llm is None or _sampling_params is None:
        raise RuntimeError("Model not initialized.")

    t0 = time.perf_counter()
    outputs_list = _llm.generate(batch_inputs, _sampling_params)
    t1 = time.perf_counter()
    if len(outputs_list) != len(batch_inputs):
        raise RuntimeError(
            f"Model returned {len(outputs_list)} outputs for {len(batch_inputs)} inputs."
        )
    results = _outputs_to_results_for_sizes(outputs_list, image_sizes)
    t2 = time.perf_counter()
    return results, {
        "gpu_generate_ms": (t1 - t0) * 1000,
        "postprocess_cpu_ms": (t2 - t1) * 1000,
    }


def _run_inference_inputs(
    inputs: list[tuple[Image.Image, str]],
    *,
    parallel_preprocess: bool = True,
) -> List[Dict[str, Any]]:
    """Run batch inference on a list of ``(image, prompt)`` inputs."""
    if _llm is None or _sampling_params is None:
        raise RuntimeError("Model not initialized.")
    if not inputs:
        return []

    t0 = time.perf_counter()
    _get_processor()
    if parallel_preprocess and len(inputs) > 1:
        with ThreadPoolExecutor(max_workers=_num_workers) as executor:
            batch_inputs = list(executor.map(_preprocess_single_with_prompt, inputs))
    else:
        batch_inputs = [_preprocess_single_with_prompt(item) for item in inputs]
    t1 = time.perf_counter()

    outputs_list = _llm.generate(batch_inputs, _sampling_params)
    if len(outputs_list) != len(inputs):
        raise RuntimeError(
            f"Model returned {len(outputs_list)} outputs for {len(inputs)} inputs."
        )
    t2 = time.perf_counter()
    preprocess_s = t1 - t0
    generate_s = t2 - t1
    total_s = t2 - t0
    prompt_tokens = sum(
        _token_count(getattr(output, "prompt_token_ids", None))
        for output in outputs_list
    )
    output_tokens = sum(_output_token_count(output) for output in outputs_list)
    prompt_tps = prompt_tokens / generate_s if generate_s > 0 else 0.0
    output_tps = output_tokens / generate_s if generate_s > 0 else 0.0
    LOGGER.info(
        (
            "inference stats: pages=%d preprocess_s=%.2f generate_s=%.2f "
            "total_s=%.2f prompt_tokens=%d output_tokens=%d "
            "prompt_tokens_per_s=%.2f output_tokens_per_s=%.2f"
        ),
        len(inputs), preprocess_s, generate_s, total_s,
        prompt_tokens, output_tokens, prompt_tps, output_tps,
    )
    return _outputs_to_results(outputs_list, [img for img, _ in inputs])


def _run_inference(images: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    """Run batch inference on a list of images for one request."""
    return _run_inference_inputs([(img, prompt) for img in images])


@dataclass
class _BatchItem:
    request_id: str
    page_index: int
    batch_input: dict
    image_size: tuple[int, int]
    future: asyncio.Future
    queued_at: float


class MicroBatcher:
    """Collect prepared pages across HTTP requests and feed one vLLM batch at a time."""

    def __init__(
        self,
        *,
        max_wait_ms: int,
        max_pages: int,
        queue_pages: int,
        generate_in_thread: bool,
    ) -> None:
        self.max_wait_s = max(0, max_wait_ms) / 1000.0
        self.max_pages = max(1, max_pages)
        self.queue: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=max(1, queue_pages))
        self._worker_task: asyncio.Task | None = None
        self._closed = False
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepseek-microbatch")
            if generate_in_thread
            else None
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Microbatcher is closed.")
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def close(self) -> None:
        self._closed = True
        if self._worker_task is None:
            self._fail_queued(RuntimeError("Microbatcher closed."))
        else:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
            self._fail_queued(RuntimeError("Microbatcher closed."))
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def _fail_items(self, items: List[_BatchItem], exc: BaseException) -> None:
        for item in items:
            if not item.future.done():
                item.future.set_exception(exc)

    def _fail_queued(self, exc: BaseException) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not item.future.done():
                item.future.set_exception(exc)

    async def submit_many(
        self,
        images: List[Image.Image],
        prompt: str,
        request_id: str,
    ) -> List[Dict[str, Any]]:
        prepared = [
            _preprocess_prepared_input((page_index, image, prompt))
            for page_index, image in enumerate(images)
        ]
        return await self.submit_prepared_many(prepared, request_id)

    async def submit_prepared_many(
        self,
        prepared_inputs: list[_PreparedInput],
        request_id: str,
    ) -> List[Dict[str, Any]]:
        self.start()
        loop = asyncio.get_running_loop()
        futures = []
        try:
            for prepared in prepared_inputs:
                future = loop.create_future()
                futures.append(future)
                await self.queue.put(
                    _BatchItem(
                        request_id=request_id,
                        page_index=prepared.page_index,
                        batch_input=prepared.batch_input,
                        image_size=prepared.image_size,
                        future=future,
                        queued_at=loop.time(),
                    )
                )
            if not futures:
                return []
            return await asyncio.gather(*futures)
        except asyncio.CancelledError:
            for future in futures:
                future.cancel()
            raise

    async def _worker_loop(self) -> None:
        while True:
            batch: List[_BatchItem] = []
            try:
                first = await self.queue.get()
                batch = [first]
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.max_wait_s

                while len(batch) < self.max_pages:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break

                await self._run_batch(batch)
            except asyncio.CancelledError as exc:
                self._fail_items(batch, RuntimeError("Microbatcher closed."))
                raise exc

    async def _run_batch(self, batch: List[_BatchItem]) -> None:
        active = [item for item in batch if not item.future.cancelled()]
        if not active:
            return

        request_count = len({item.request_id for item in active})
        now = asyncio.get_running_loop().time()
        waits_ms = sorted((now - item.queued_at) * 1000 for item in active)
        wait_p50_ms = waits_ms[len(waits_ms) // 2]
        wait_max_ms = waits_ms[-1]

        try:
            started = time.perf_counter()
            batch_id = uuid4().hex[:12]
            batch_inputs = [item.batch_input for item in active]
            image_sizes = [item.image_size for item in active]
            run_batch = partial(
                _run_prepared_batch_sync,
                batch_inputs,
                image_sizes,
            )
            if self._executor is None:
                results, batch_timing = run_batch()
            else:
                results, batch_timing = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    run_batch,
                )
            elapsed = time.perf_counter() - started
        except Exception as exc:
            LOGGER.exception(
                "microbatch failed: pages=%d requests=%d",
                len(active), request_count,
            )
            for item in active:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        LOGGER.info(
            (
                "microbatch complete: pages=%d requests=%d wait_p50_ms=%.1f "
                "wait_max_ms=%.1f queue_pages_after_start=%d total_s=%.2f"
            ),
            len(active), request_count, wait_p50_ms, wait_max_ms,
            self.queue.qsize(), elapsed,
        )
        for item, result in zip(active, results):
            if not item.future.done():
                result["_batch_id"] = batch_id
                result["_queue_wait_ms"] = (now - item.queued_at) * 1000
                result["_batch_gpu_generate_ms"] = batch_timing["gpu_generate_ms"]
                result["_batch_postprocess_cpu_ms"] = batch_timing["postprocess_cpu_ms"]
                item.future.set_result(result)


def _record_batch_timing_from_results(
    timer: StepTimer,
    results: List[Dict[str, Any]],
) -> None:
    max_queue_wait_ms = 0.0
    gpu_generate_by_batch: dict[str, float] = {}
    postprocess_by_batch: dict[str, float] = {}
    for result in results:
        max_queue_wait_ms = max(
            max_queue_wait_ms,
            float(result.pop("_queue_wait_ms", 0.0) or 0.0),
        )
        batch_id = result.pop("_batch_id", None)
        gpu_ms = float(result.pop("_batch_gpu_generate_ms", 0.0) or 0.0)
        post_ms = float(result.pop("_batch_postprocess_cpu_ms", 0.0) or 0.0)
        if batch_id is not None:
            gpu_generate_by_batch.setdefault(str(batch_id), gpu_ms)
            postprocess_by_batch.setdefault(str(batch_id), post_ms)

    if max_queue_wait_ms:
        timer.record("gpu_queue_wait", max_queue_wait_ms)
    for value in gpu_generate_by_batch.values():
        timer.record("gpu_generate", value)
    for value in postprocess_by_batch.values():
        timer.record("postprocess_cpu", value)
    timer.record_meta("gpu_batches", len(gpu_generate_by_batch))


async def _run_prepared_inference_for_request(
    prepared_inputs: list[_PreparedInput],
    request_id: str,
    timer: StepTimer,
) -> List[Dict[str, Any]]:
    if not prepared_inputs:
        timer.record_meta("gpu_batches", 0)
        return []

    if not _microbatch_enabled:
        results, batch_timing = _run_prepared_batch_sync(
            [item.batch_input for item in prepared_inputs],
            [item.image_size for item in prepared_inputs],
        )
        timer.record("gpu_generate", batch_timing["gpu_generate_ms"])
        timer.record("postprocess_cpu", batch_timing["postprocess_cpu_ms"])
        timer.record_meta("gpu_batches", 1 if prepared_inputs else 0)
        return results
    if _microbatcher is None:
        raise RuntimeError("Microbatcher is not initialized.")
    results = await _microbatcher.submit_prepared_many(prepared_inputs, request_id)
    _record_batch_timing_from_results(timer, results)
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    queue_pages = _microbatcher.queue.qsize() if _microbatcher is not None else 0
    return {
        "status": "ok",
        "model_loaded": _llm is not None,
        "microbatch_queue_pages": queue_pages,
        "render_workers": _render_workers,
        "preprocess_workers": _num_workers,
    }


@app.post("/predict/image")
async def predict_image(request: Request, prompt: Optional[str] = None):
    """OCR a single image.

    Send raw image bytes as the request body.
    Optionally pass ?prompt=... query param to override the default prompt.
    """
    timer = StepTimer()
    with timer.step("request_body"):
        image_bytes = await request.body()
    if not image_bytes:
        return _timed_json_response(
            {"detail": "Empty request body. Send raw image bytes."},
            timer,
            status_code=400,
        )

    if prompt is None:
        prompt = DEFAULT_PROMPT

    req_id = _request_id()
    started = time.perf_counter()
    LOGGER.info(
        "request start: id=%s path=/predict/image bytes=%d",
        req_id, len(image_bytes),
    )
    try:
        loop = asyncio.get_running_loop()
        with timer.step("image_decode_cpu"):
            img = await loop.run_in_executor(
                _get_render_executor(),
                _load_image_bytes,
                image_bytes,
            )
        timer.record_meta("page_count", 1)
        prepared = await _prepare_images_for_inference([img], prompt, timer)
        results = await _run_prepared_inference_for_request(prepared, req_id, timer)
        LOGGER.info(
            "request complete: id=%s path=/predict/image images=1 total_s=%.2f",
            req_id, time.perf_counter() - started,
        )
        usage = _response_usage_from_results(results)
        return _timed_json_response({"results": results, "usage": usage}, timer)
    except Exception as exc:
        LOGGER.exception("Inference failed")
        return _timed_json_response({"detail": str(exc)}, timer, status_code=500)


@app.post("/predict/pdf")
async def predict_pdf(request: Request, prompt: Optional[str] = None, dpi: int = 144):
    """OCR all pages of a PDF.

    Send raw PDF bytes as the request body.
    Query params:
        prompt: override default prompt (default: Free OCR)
        dpi: render resolution (default: 144)
    """
    timer = StepTimer()
    with timer.step("request_body"):
        pdf_bytes = await request.body()
    if not pdf_bytes:
        return _timed_json_response(
            {"detail": "Empty request body. Send raw PDF bytes."},
            timer,
            status_code=400,
        )

    if prompt is None:
        prompt = DEFAULT_PROMPT

    req_id = _request_id()
    started = time.perf_counter()
    LOGGER.info(
        "request start: id=%s path=/predict/pdf bytes=%d dpi=%d",
        req_id, len(pdf_bytes), dpi,
    )
    try:
        loop = asyncio.get_running_loop()
        with timer.step("pdf_render_cpu"):
            images = await loop.run_in_executor(
                _get_render_executor(),
                partial(_pdf_to_images, pdf_bytes, dpi=dpi),
            )
        render_s = timer.summary()["steps"].get("pdf_render_cpu", 0) / 1000
        LOGGER.info(
            "pdf render: id=%s pages=%d render_s=%.2f",
            req_id, len(images), render_s,
        )
        timer.record_meta("page_count", len(images))
        prepared = await _prepare_images_for_inference(images, prompt, timer)
        results = await _run_prepared_inference_for_request(prepared, req_id, timer)
        LOGGER.info(
            "request complete: id=%s path=/predict/pdf pages=%d total_s=%.2f",
            req_id, len(prepared), time.perf_counter() - started,
        )
        usage = _response_usage_from_results(results)
        return _timed_json_response(
            {"page_count": len(prepared), "results": results, "usage": usage},
            timer,
        )
    except Exception as exc:
        LOGGER.exception("PDF inference failed")
        return _timed_json_response({"detail": str(exc)}, timer, status_code=500)


@app.post("/predict/batch")
async def predict_batch(request: Request):
    """OCR a batch of images.

    JSON body:
    {
        "images": ["<base64-encoded-image>", ...],
        "prompt": "<optional prompt override>"
    }
    """
    timer = StepTimer()
    try:
        with timer.step("request_body"):
            body = await request.json()
    except Exception:
        return _timed_json_response({"detail": "Invalid JSON body."}, timer, status_code=400)

    images_b64 = body.get("images", [])
    prompt = body.get("prompt", DEFAULT_PROMPT)

    if not images_b64:
        return _timed_json_response({"detail": "No images provided."}, timer, status_code=400)

    req_id = _request_id()
    started = time.perf_counter()
    LOGGER.info(
        "request start: id=%s path=/predict/batch images=%d",
        req_id, len(images_b64),
    )
    try:
        loop = asyncio.get_running_loop()

        def load_batch_images() -> list[Image.Image]:
            loaded = []
            for b64_str in images_b64:
                img_bytes = base64.b64decode(b64_str)
                loaded.append(_load_image_bytes(img_bytes))
            return loaded

        with timer.step("image_decode_cpu"):
            pil_images = await loop.run_in_executor(_get_render_executor(), load_batch_images)
        timer.record_meta("page_count", len(pil_images))
        prepared = await _prepare_images_for_inference(pil_images, prompt, timer)
        results = await _run_prepared_inference_for_request(prepared, req_id, timer)
        LOGGER.info(
            "request complete: id=%s path=/predict/batch images=%d total_s=%.2f",
            req_id, len(prepared), time.perf_counter() - started,
        )
        usage = _response_usage_from_results(results)
        return _timed_json_response({"results": results, "usage": usage}, timer)
    except Exception as exc:
        LOGGER.exception("Batch inference failed")
        return _timed_json_response({"detail": str(exc)}, timer, status_code=500)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR-2 model service (vLLM)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8300, help="Bind port.")
    parser.add_argument("--model-path", required=True, help="HuggingFace model ID or local path.")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEM_UTIL)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cuda-devices", default=None, help="CUDA_VISIBLE_DEVICES override.")
    parser.add_argument("--crop-mode", action="store_true", default=True, help="Enable image cropping (default: on).")
    parser.add_argument("--no-crop-mode", dest="crop_mode", action="store_false", help="Disable image cropping.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Image preprocessing workers.")
    parser.add_argument(
        "--render-workers",
        type=int,
        default=DEFAULT_RENDER_WORKERS,
        help=f"PDF/image decode workers (default: {DEFAULT_RENDER_WORKERS}).",
    )
    parser.add_argument(
        "--microbatch-max-wait-ms",
        type=int,
        default=DEFAULT_MICROBATCH_MAX_WAIT_MS,
        help="Maximum time to wait for more pages after the first queued page (default: 150).",
    )
    parser.add_argument(
        "--microbatch-max-pages",
        type=int,
        default=DEFAULT_MICROBATCH_MAX_PAGES,
        help="Maximum pages per cross-request GPU batch (default: 64).",
    )
    parser.add_argument(
        "--microbatch-queue-pages",
        type=int,
        default=DEFAULT_MICROBATCH_QUEUE_PAGES,
        help="Maximum queued pages waiting for GPU microbatching (default: 512).",
    )
    parser.add_argument(
        "--disable-microbatching",
        action="store_true",
        default=False,
        help="Disable cross-request microbatching and use the previous per-request inference path.",
    )
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable CUDA graph capture (slower decode, less VRAM, faster startup).")
    args = parser.parse_args()
    if not args.model_path.strip():
        parser.error("--model-path must not be empty.")
    if args.num_workers < 1:
        parser.error("--num-workers must be greater than or equal to 1.")
    if args.render_workers < 1:
        parser.error("--render-workers must be greater than or equal to 1.")
    if args.microbatch_max_wait_ms < 0:
        parser.error("--microbatch-max-wait-ms must be greater than or equal to 0.")
    if args.microbatch_max_pages < 1:
        parser.error("--microbatch-max-pages must be greater than or equal to 1.")
    if args.microbatch_queue_pages < 1:
        parser.error("--microbatch-queue-pages must be greater than or equal to 1.")
    return args


def main() -> None:
    args = parse_args()

    global _crop_mode, _microbatch_enabled, _microbatcher, _num_workers, _render_workers  # noqa: PLW0603
    _crop_mode = args.crop_mode
    _num_workers = args.num_workers
    _render_workers = args.render_workers
    _microbatch_enabled = not args.disable_microbatching
    _microbatcher = MicroBatcher(
        max_wait_ms=args.microbatch_max_wait_ms,
        max_pages=args.microbatch_max_pages,
        queue_pages=args.microbatch_queue_pages,
        generate_in_thread=True,
    )

    _init_model(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        cuda_devices=args.cuda_devices,
        enforce_eager=args.enforce_eager,
    )
    LOGGER.info(
        "Microbatching %s (max_wait_ms=%d, max_pages=%d, queue_pages=%d)",
        "enabled" if _microbatch_enabled else "disabled",
        args.microbatch_max_wait_ms,
        args.microbatch_max_pages,
        args.microbatch_queue_pages,
    )
    LOGGER.info(
        "CPU workers: render_workers=%d preprocess_workers=%d",
        args.render_workers,
        args.num_workers,
    )
    if _microbatch_enabled:
        LOGGER.info("Microbatch inference runs in a dedicated executor thread")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
