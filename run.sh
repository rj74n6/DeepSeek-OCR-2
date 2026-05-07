#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SERVICE_DIR="${SCRIPT_DIR}/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm"
CALLER_DIR="$(pwd -P)"

resolve_model_path() {
  local model_path="$1"

  case "${model_path}" in
    /*)
      printf '%s\n' "${model_path}"
      ;;
    ./*|../*)
      printf '%s\n' "$(cd "${CALLER_DIR}" && cd "$(dirname "${model_path}")" && pwd -P)/$(basename "${model_path}")"
      ;;
    *)
      printf '%s\n' "${model_path}"
      ;;
  esac
}

HAS_MODEL_PATH=0
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  case "${ARGS[$i]}" in
    -h|--help)
      cd "${SERVICE_DIR}"
      exec uv run python start_service.py "${ARGS[@]}"
      ;;
    --model-path)
      HAS_MODEL_PATH=1
      next_index=$((i + 1))
      if [ "${next_index}" -ge "${#ARGS[@]}" ]; then
        echo "error: --model-path requires a value." >&2
        exit 2
      fi
      ARGS[$next_index]="$(resolve_model_path "${ARGS[$next_index]}")"
      ;;
    --model-path=*)
      HAS_MODEL_PATH=1
      ARGS[$i]="--model-path=$(resolve_model_path "${ARGS[$i]#--model-path=}")"
      ;;
  esac
done

if [ "${HAS_MODEL_PATH}" -ne 1 ]; then
  echo "error: --model-path is required." >&2
  echo "usage: $0 --model-path /path/to/DeepSeek-OCR-2 [service args...]" >&2
  exit 2
fi

cd "${SERVICE_DIR}"
if [ -x "${SCRIPT_DIR}/.venv/bin/python" ]; then
  exec "${SCRIPT_DIR}/.venv/bin/python" start_service.py "${ARGS[@]}"
fi

exec uv run python start_service.py "${ARGS[@]}"
