#!/usr/bin/env python3
"""Offline DeepSeek-OCR-2 batch runner.

This bypasses the HTTP service so offline jobs can measure and use the
largest efficient vLLM page batches directly.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "DeepSeek-OCR2-master" / "DeepSeek-OCR2-vllm"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import start_service as service  # noqa: E402


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
PROCESSOR: Any | None = None


def _get_processor() -> Any:
    global PROCESSOR  # noqa: PLW0603
    if PROCESSOR is None:
        from process.image_process import DeepseekOCR2Processor

        PROCESSOR = DeepseekOCR2Processor()
    return PROCESSOR


def _preprocess_page_input(item: tuple[Image.Image, str, bool]) -> dict[str, Any]:
    image, prompt, crop_mode = item
    return {
        "prompt": prompt,
        "multi_modal_data": {
            "image": _get_processor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=crop_mode,
                prompt=prompt,
            )
        },
    }


def _worker_pid(_: int) -> int:
    return os.getpid()


@dataclass
class RunnerConfig:
    input_folder: Path
    model_path: str
    output_jsonl: Path
    metrics_json: Path
    batch_pages: int = service.DEFAULT_MICROBATCH_MAX_PAGES
    limit_docs: int | None = None
    dpi: int = 144
    prompt: str = service.DEFAULT_PROMPT
    preprocess_executor: str = "thread"
    num_workers: int = service.DEFAULT_NUM_WORKERS
    crop_mode: bool = service.DEFAULT_CROP_MODE
    max_model_len: int = service.DEFAULT_MAX_MODEL_LEN
    max_num_seqs: int = service.DEFAULT_MAX_CONCURRENCY
    gpu_memory_utilization: float = service.DEFAULT_GPU_MEM_UTIL
    tensor_parallel_size: int = 1
    cuda_devices: str | None = None
    enforce_eager: bool = False


@dataclass
class DocumentState:
    index: int
    path: Path
    started_at: float = field(default_factory=time.perf_counter)
    page_count: int = 0
    render_s: float = 0.0
    results: list[dict[str, Any] | None] = field(default_factory=list)
    failed: bool = False
    error: str | None = None

    def set_pages(self, page_count: int, render_s: float) -> None:
        self.page_count = page_count
        self.render_s = render_s
        self.results = [None] * page_count

    def mark_failed(self, error: str) -> None:
        self.failed = True
        self.error = error

    @property
    def complete(self) -> bool:
        return self.failed or (
            bool(self.results)
            and all(result is not None for result in self.results)
        )

    def to_json(self) -> dict[str, Any]:
        timings = {
            "render_s": round(self.render_s, 6),
            "total_s": round(time.perf_counter() - self.started_at, 6),
        }
        if self.failed:
            return {
                "document_index": self.index,
                "document_path": str(self.path),
                "status": "failed",
                "error": self.error or "unknown error",
                "page_count": self.page_count,
                "results": [],
                "usage": _zero_usage(),
                "timings": timings,
            }

        results = [result for result in self.results if result is not None]
        usage = service._response_usage_from_results(results)
        return {
            "document_index": self.index,
            "document_path": str(self.path),
            "status": "ok",
            "page_count": self.page_count,
            "results": results,
            "usage": usage,
            "timings": timings,
        }


@dataclass
class PageJob:
    document_index: int
    page_index: int
    image: Image.Image
    prompt: str


class OfflineOcrRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.states: dict[int, DocumentState] = {}
        self.pending_pages: list[PageJob] = []
        self.next_write_index = 0
        self.metrics: dict[str, Any] = {
            "documents": 0,
            "succeeded": 0,
            "failed": 0,
            "pages": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "generated_pages": 0,
            "generated_input_tokens": 0,
            "generated_cached_tokens": 0,
            "generated_output_tokens": 0,
            "generated_total_tokens": 0,
            "render_s": 0.0,
            "preprocess_s": 0.0,
            "generate_s": 0.0,
            "inference_s": 0.0,
            "batches": [],
            "failures": [],
        }

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        self.config.metrics_json.parent.mkdir(parents=True, exist_ok=True)

        with self._preprocess_pool() as pool:
            self._pool = pool
            self._warm_preprocess_pool()
            self._init_model()
            with self.config.output_jsonl.open("w", encoding="utf-8") as output:
                for doc_index, path in enumerate(
                    discover_documents(self.config.input_folder, self.config.limit_docs)
                ):
                    self.metrics["documents"] += 1
                    self._render_enqueue_document(doc_index, path)
                    while len(self.pending_pages) >= self.config.batch_pages:
                        self._flush_next_batch()
                        self._write_ready_documents(output)

                while self.pending_pages:
                    self._flush_next_batch()
                    self._write_ready_documents(output)

                self._write_ready_documents(output)

        elapsed_s = time.perf_counter() - started
        self.metrics["elapsed_s"] = round(elapsed_s, 6)
        self.metrics["pages_per_second"] = (
            self.metrics["pages"] / elapsed_s if elapsed_s > 0 else 0.0
        )
        self.metrics["tokens_per_second"] = (
            self.metrics["total_tokens"] / elapsed_s if elapsed_s > 0 else 0.0
        )
        self.metrics["generated_pages_per_second"] = (
            self.metrics["generated_pages"] / elapsed_s if elapsed_s > 0 else 0.0
        )
        self.metrics["generated_tokens_per_second"] = (
            self.metrics["generated_total_tokens"] / elapsed_s if elapsed_s > 0 else 0.0
        )
        self.metrics["config"] = {
            "batch_pages": self.config.batch_pages,
            "preprocess_executor": self.config.preprocess_executor,
            "num_workers": self.config.num_workers,
            "dpi": self.config.dpi,
            "crop_mode": self.config.crop_mode,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "tensor_parallel_size": self.config.tensor_parallel_size,
        }
        self._write_metrics()
        return self.metrics

    def _init_model(self) -> None:
        service._crop_mode = self.config.crop_mode
        service._num_workers = self.config.num_workers
        service._init_model(
            model_path=self.config.model_path,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            tensor_parallel_size=self.config.tensor_parallel_size,
            cuda_devices=self.config.cuda_devices,
            enforce_eager=self.config.enforce_eager,
        )

    def _preprocess_pool(self):
        workers = max(1, self.config.num_workers)
        if self.config.preprocess_executor == "serial":
            return _NullPool()
        if self.config.preprocess_executor == "process":
            return ProcessPoolExecutor(max_workers=workers)
        return ThreadPoolExecutor(max_workers=workers)

    def _render_enqueue_document(self, doc_index: int, path: Path) -> None:
        state = DocumentState(index=doc_index, path=path)
        self.states[doc_index] = state
        try:
            t0 = time.perf_counter()
            images = _render_document(path, dpi=self.config.dpi)
            render_s = time.perf_counter() - t0
            state.set_pages(len(images), render_s)
            self.metrics["render_s"] += render_s
            for page_index, image in enumerate(images):
                self.pending_pages.append(
                    PageJob(
                        document_index=doc_index,
                        page_index=page_index,
                        image=image,
                        prompt=self.config.prompt,
                    )
                )
            if not images:
                state.mark_failed("document produced zero pages")
                self._record_failure(state)
        except Exception as exc:
            state.mark_failed(str(exc))
            self._record_failure(state)

    def _flush_next_batch(self) -> None:
        batch = self.pending_pages[: self.config.batch_pages]
        self.pending_pages = self.pending_pages[self.config.batch_pages :]
        active = [
            page
            for page in batch
            if not self.states[page.document_index].failed
        ]
        active_ids = {id(page) for page in active}
        self._close_pages([page for page in batch if id(page) not in active_ids])
        if not active:
            return

        try:
            batch_metrics, results = self._run_batch(active)
        except Exception as exc:
            failed_doc_indexes = sorted({page.document_index for page in active})
            for doc_index in failed_doc_indexes:
                state = self.states[doc_index]
                if not state.failed:
                    state.mark_failed(str(exc))
                    self._record_failure(state)
            dropped_pages = [
                page for page in self.pending_pages
                if page.document_index in failed_doc_indexes
            ]
            self.pending_pages = [
                page for page in self.pending_pages
                if page.document_index not in failed_doc_indexes
            ]
            self._close_pages(active)
            self._close_pages(dropped_pages)
            return

        for page, result in zip(active, results):
            self.states[page.document_index].results[page.page_index] = result
        self._close_pages(active)
        self._record_batch_metrics(batch_metrics, results, active)

    def _run_batch(
        self,
        pages: list[PageJob],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if service._llm is None or service._sampling_params is None:
            raise RuntimeError("Model not initialized.")

        t0 = time.perf_counter()
        preprocess_items = [
            (page.image, page.prompt, self.config.crop_mode)
            for page in pages
        ]
        batch_inputs = list(self._pool.map(_preprocess_page_input, preprocess_items))
        t1 = time.perf_counter()

        outputs = service._llm.generate(batch_inputs, service._sampling_params)
        if len(outputs) != len(pages):
            raise RuntimeError(
                f"Model returned {len(outputs)} outputs for {len(pages)} inputs."
            )
        t2 = time.perf_counter()

        results = service._outputs_to_results(outputs, [page.image for page in pages])
        return (
            {
                "pages": len(pages),
                "documents": len({page.document_index for page in pages}),
                "preprocess_s": t1 - t0,
                "generate_s": t2 - t1,
                "total_s": t2 - t0,
            },
            results,
        )

    def _record_batch_metrics(
        self,
        batch_metrics: dict[str, Any],
        results: list[dict[str, Any]],
        pages: list[PageJob],
    ) -> None:
        usage = _usage_from_results_copy(results)
        self.metrics["generated_pages"] += len(pages)
        self.metrics["generated_input_tokens"] += usage["input_tokens"]
        self.metrics["generated_cached_tokens"] += usage["cached_tokens"]
        self.metrics["generated_output_tokens"] += usage["output_tokens"]
        self.metrics["generated_total_tokens"] += usage["total_tokens"]
        self.metrics["preprocess_s"] += batch_metrics["preprocess_s"]
        self.metrics["generate_s"] += batch_metrics["generate_s"]
        self.metrics["inference_s"] += batch_metrics["total_s"]
        self.metrics["batches"].append({
            "pages": batch_metrics["pages"],
            "documents": batch_metrics["documents"],
            "preprocess_s": round(batch_metrics["preprocess_s"], 6),
            "generate_s": round(batch_metrics["generate_s"], 6),
            "total_s": round(batch_metrics["total_s"], 6),
            "pages_per_second": (
                batch_metrics["pages"] / batch_metrics["total_s"]
                if batch_metrics["total_s"] > 0
                else 0.0
            ),
        })

    def _write_ready_documents(self, output) -> None:
        while True:
            state = self.states.get(self.next_write_index)
            if state is None or not state.complete:
                return
            payload = state.to_json()
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
            if payload["status"] == "ok":
                self.metrics["succeeded"] += 1
                usage = payload["usage"]
                self.metrics["pages"] += int(payload["page_count"])
                self.metrics["input_tokens"] += int(usage["input_tokens"])
                self.metrics["cached_tokens"] += int(usage["cached_tokens"])
                self.metrics["output_tokens"] += int(usage["output_tokens"])
                self.metrics["total_tokens"] += int(usage["total_tokens"])
            else:
                self.metrics["failed"] += 1
            del self.states[self.next_write_index]
            self.next_write_index += 1

    def _record_failure(self, state: DocumentState) -> None:
        self.metrics["failures"].append({
            "document_index": state.index,
            "document_path": str(state.path),
            "error": state.error or "unknown error",
        })

    def _write_metrics(self) -> None:
        for key in ("render_s", "preprocess_s", "generate_s", "inference_s"):
            self.metrics[key] = round(self.metrics[key], 6)
        with self.config.metrics_json.open("w", encoding="utf-8") as output:
            json.dump(self.metrics, output, indent=2, sort_keys=True)
            output.write("\n")

    @staticmethod
    def _close_pages(pages: Iterable[PageJob]) -> None:
        for page in pages:
            with contextlib.suppress(Exception):
                page.image.close()

    def _warm_preprocess_pool(self) -> None:
        if self.config.preprocess_executor != "process":
            return
        list(self._pool.map(_worker_pid, range(max(1, self.config.num_workers))))


class _NullPool:
    def __enter__(self) -> "_NullPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def map(self, func, iterable):
        return map(func, iterable)


def discover_documents(folder: Path, limit_docs: int | None = None) -> list[Path]:
    paths = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    paths.sort()
    if limit_docs is not None:
        return paths[:limit_docs]
    return paths


def _render_document(path: Path, *, dpi: int) -> list[Image.Image]:
    if path.suffix.lower() == ".pdf":
        return service._pdf_to_images(path.read_bytes(), dpi=dpi)
    return [service._load_image_bytes(path.read_bytes())]


def _usage_from_results_copy(results: list[dict[str, Any]]) -> dict[str, int]:
    usage = _zero_usage()
    for result in results:
        item_usage = result.get("_usage") or {}
        usage["input_tokens"] += int(item_usage.get("input_tokens", 0) or 0)
        usage["cached_tokens"] += int(item_usage.get("cached_tokens", 0) or 0)
        usage["output_tokens"] += int(item_usage.get("output_tokens", 0) or 0)
        usage["total_tokens"] += int(item_usage.get("total_tokens", 0) or 0)
    return usage


def _zero_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "input_tokens_details": {"cached_tokens": 0},
    }


def parse_args(argv: list[str] | None = None) -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Offline DeepSeek-OCR-2 batch runner")
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--batch-pages", type=int, default=service.DEFAULT_MICROBATCH_MAX_PAGES)
    parser.add_argument("--limit-docs", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--prompt", default=service.DEFAULT_PROMPT)
    parser.add_argument(
        "--preprocess-executor",
        choices=("thread", "process", "serial"),
        default="thread",
    )
    parser.add_argument("--num-workers", type=int, default=service.DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-model-len", type=int, default=service.DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=service.DEFAULT_MAX_CONCURRENCY)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=service.DEFAULT_GPU_MEM_UTIL,
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--crop-mode", action="store_true", default=True)
    parser.add_argument("--no-crop-mode", dest="crop_mode", action="store_false")
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    args = parser.parse_args(argv)

    if args.batch_pages < 1:
        parser.error("--batch-pages must be greater than or equal to 1.")
    if args.limit_docs is not None and args.limit_docs < 1:
        parser.error("--limit-docs must be greater than or equal to 1.")
    if args.num_workers < 1:
        parser.error("--num-workers must be greater than or equal to 1.")
    if not args.input_folder.is_dir():
        parser.error(f"{args.input_folder} is not a directory.")

    return RunnerConfig(
        input_folder=args.input_folder,
        model_path=args.model_path,
        output_jsonl=args.output_jsonl,
        metrics_json=args.metrics_json,
        batch_pages=args.batch_pages,
        limit_docs=args.limit_docs,
        dpi=args.dpi,
        prompt=args.prompt,
        preprocess_executor=args.preprocess_executor,
        num_workers=args.num_workers,
        crop_mode=args.crop_mode,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        cuda_devices=args.cuda_devices,
        enforce_eager=args.enforce_eager,
    )


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    metrics = OfflineOcrRunner(config).run()
    print(json.dumps({
        "documents": metrics["documents"],
        "succeeded": metrics["succeeded"],
        "failed": metrics["failed"],
        "pages": metrics["pages"],
        "pages_per_second": metrics["pages_per_second"],
        "generated_pages": metrics["generated_pages"],
        "generated_pages_per_second": metrics["generated_pages_per_second"],
        "tokens_per_second": metrics["tokens_per_second"],
        "metrics_json": str(config.metrics_json),
        "output_jsonl": str(config.output_jsonl),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
