# DeepSeek-OCR-2 vLLM Service

This directory packages DeepSeek-OCR-2 as a local HTTP service for the KIE
backend. Users should only need the two root scripts:

- `install.sh` installs the Python/vLLM dependencies with `uv`.
- `run.sh` starts the FastAPI service.

The service implementation lives under
`DeepSeek-OCR2-master/DeepSeek-OCR2-vllm/start_service.py` so it can import the
DeepSeek vLLM modules directly.

## Install

Run from this directory:

```bash
./install.sh
```

The installer expects `uv` to be available and installs the dependencies listed
in `requirements.txt`, plus PyTorch, vLLM, and flash-attn.

## Run

Start the OCR service:

```bash
./run.sh --model-path /path/to/DeepSeek-OCR-2 --port 8888
```

Optional arguments are forwarded to `start_service.py`:

```bash
./run.sh \
  --model-path /path/to/DeepSeek-OCR-2 \
  --gpu-memory-utilization 0.85 \
  --cuda-devices 0 \
  --port 8888
```

Point the KIE backend at the service:

```bash
DEEPSEEK_OCR_ENDPOINT=http://localhost:8888
```

## Check

```bash
curl http://localhost:8888/health
```

Expected shape:

```json
{"status":"ok","model_loaded":true}
```

## Endpoints

- `POST /predict/image`: raw image bytes in the request body.
- `POST /predict/pdf`: raw PDF bytes in the request body.
- `POST /predict/batch`: JSON body with base64 images:

```json
{"images":["<base64-image>"],"prompt":"<optional prompt>"}
```

## Source

DeepSeek upstream project links:

- Model: https://huggingface.co/deepseek-ai/DeepSeek-OCR-2
- Paper: https://arxiv.org/abs/2601.20552
