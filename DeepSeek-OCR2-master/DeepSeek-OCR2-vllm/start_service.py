"""DeepSeek-OCR-2 model service based on vLLM.

Usage (conda env: deepseek-ocr2):
    python start_service.py --model-path /path/to/model --port 8300
    python start_service.py --model-path /path/to/model --gpu-memory-utilization 0.85
"""

import argparse
import base64
import io
import logging
import os
import re
import time
from copy import deepcopy
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import torch

if torch.version.cuda == "11.8":
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ["VLLM_USE_V1"] = "0"

import fitz
import uvicorn
from fastapi import FastAPI, HTTPException, Request
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

from deepseek_ocr2 import DeepseekOCR2ForCausalLM  # noqa: E402
from process.image_process import DeepseekOCR2Processor  # noqa: E402
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.model_executor.models.registry import ModelRegistry  # noqa: E402

ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_MODEL_LEN = 8192
DEFAULT_MAX_CONCURRENCY = 100
DEFAULT_GPU_MEM_UTIL = 0.3
DEFAULT_CROP_MODE = True
DEFAULT_NUM_WORKERS = 64
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
app = FastAPI(title="DeepSeek-OCR-2 Service")
_llm: Optional[LLM] = None
_sampling_params: Optional[SamplingParams] = None
_crop_mode: bool = DEFAULT_CROP_MODE
_num_workers: int = DEFAULT_NUM_WORKERS
_processor: DeepseekOCR2Processor = DeepseekOCR2Processor()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        images.append(img)
    doc.close()
    return images


def _load_image_bytes(image_bytes: bytes) -> Image.Image:
    """Load an image from raw bytes, applying EXIF orientation correction."""
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _preprocess_single(image: Image.Image) -> dict:
    """Tokenize a single image for the model."""
    prompt = DEFAULT_PROMPT
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": _processor.tokenize_with_images(
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
            "image": _processor.tokenize_with_images(
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


def _run_inference(images: List[Image.Image], prompt: str) -> List[Dict[str, Any]]:
    """Run batch inference on a list of images."""
    if _llm is None or _sampling_params is None:
        raise RuntimeError("Model not initialized.")

    args_list = [(img, prompt) for img in images]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=_num_workers) as executor:
        batch_inputs = list(executor.map(_preprocess_single_with_prompt, args_list))
    t1 = time.perf_counter()

    outputs_list = _llm.generate(batch_inputs, _sampling_params)
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
        len(images), preprocess_s, generate_s, total_s,
        prompt_tokens, output_tokens, prompt_tps, output_tps,
    )

    results = []
    for output, img in zip(outputs_list, images):
        raw_text = output.outputs[0].text
        w, h = img.size
        has_eos = "<｜end▁of▁sentence｜>" in raw_text
        results.append({
            "raw_text": raw_text,
            "clean_text": _clean_output(raw_text),
            "detections": _extract_detections(raw_text, w, h),
            "has_eos": has_eos,
            "image_size": {"width": w, "height": h},
        })
    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": _llm is not None}


@app.post("/predict/image")
async def predict_image(request: Request, prompt: Optional[str] = None):
    """OCR a single image.

    Send raw image bytes as the request body.
    Optionally pass ?prompt=... query param to override the default prompt.
    """
    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty request body. Send raw image bytes.")

    if prompt is None:
        prompt = DEFAULT_PROMPT

    req_id = _request_id()
    started = time.perf_counter()
    LOGGER.info(
        "request start: id=%s path=/predict/image bytes=%d",
        req_id, len(image_bytes),
    )
    try:
        img = _load_image_bytes(image_bytes)
        results = _run_inference([img], prompt)
        LOGGER.info(
            "request complete: id=%s path=/predict/image images=1 total_s=%.2f",
            req_id, time.perf_counter() - started,
        )
        return JSONResponse({"results": results})
    except Exception as exc:
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/pdf")
async def predict_pdf(request: Request, prompt: Optional[str] = None, dpi: int = 144):
    """OCR all pages of a PDF.

    Send raw PDF bytes as the request body.
    Query params:
        prompt: override default prompt (default: Free OCR)
        dpi: render resolution (default: 144)
    """
    pdf_bytes = await request.body()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty request body. Send raw PDF bytes.")

    if prompt is None:
        prompt = DEFAULT_PROMPT

    req_id = _request_id()
    started = time.perf_counter()
    LOGGER.info(
        "request start: id=%s path=/predict/pdf bytes=%d dpi=%d",
        req_id, len(pdf_bytes), dpi,
    )
    try:
        t0 = time.perf_counter()
        images = _pdf_to_images(pdf_bytes, dpi=dpi)
        render_s = time.perf_counter() - t0
        LOGGER.info(
            "pdf render: id=%s pages=%d render_s=%.2f",
            req_id, len(images), render_s,
        )
        results = _run_inference(images, prompt)
        LOGGER.info(
            "request complete: id=%s path=/predict/pdf pages=%d total_s=%.2f",
            req_id, len(images), time.perf_counter() - started,
        )
        return JSONResponse({"page_count": len(images), "results": results})
    except Exception as exc:
        LOGGER.exception("PDF inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict/batch")
async def predict_batch(request: Request):
    """OCR a batch of images.

    JSON body:
    {
        "images": ["<base64-encoded-image>", ...],
        "prompt": "<optional prompt override>"
    }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    images_b64 = body.get("images", [])
    prompt = body.get("prompt", DEFAULT_PROMPT)

    if not images_b64:
        raise HTTPException(status_code=400, detail="No images provided.")

    req_id = _request_id()
    started = time.perf_counter()
    LOGGER.info(
        "request start: id=%s path=/predict/batch images=%d",
        req_id, len(images_b64),
    )
    try:
        pil_images = []
        for b64_str in images_b64:
            img_bytes = base64.b64decode(b64_str)
            pil_images.append(_load_image_bytes(img_bytes))
        results = _run_inference(pil_images, prompt)
        LOGGER.info(
            "request complete: id=%s path=/predict/batch images=%d total_s=%.2f",
            req_id, len(pil_images), time.perf_counter() - started,
        )
        return JSONResponse({"results": results})
    except Exception as exc:
        LOGGER.exception("Batch inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable CUDA graph capture (slower decode, less VRAM, faster startup).")
    args = parser.parse_args()
    if not args.model_path.strip():
        parser.error("--model-path must not be empty.")
    return args


def main() -> None:
    args = parse_args()

    global _crop_mode, _num_workers  # noqa: PLW0603
    _crop_mode = args.crop_mode
    _num_workers = args.num_workers

    _init_model(
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        cuda_devices=args.cuda_devices,
        enforce_eager=args.enforce_eager,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=_uvicorn_log_config(),
    )


if __name__ == "__main__":
    main()
