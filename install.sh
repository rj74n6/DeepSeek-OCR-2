#!/bin/bash

# This script sets up the environment for DeepSeek by installing the necessary dependencies.
# It uses the 'uv' tool to manage Python versions and dependencies
# Run the script in the root directory of the DeepSeek project.

set -euo pipefail 

TOTAL_STEPS=7
CURRENT_STEP=0

step() {
  CURRENT_STEP=$((CURRENT_STEP + 1))
  echo "[$CURRENT_STEP/$TOTAL_STEPS] $1"
}

if [ ! -f requirements.txt ]; then
  echo "error: requirements.txt not found. Run this from the DeepSeek project root." >&2
  exit 1
fi

# Ensure Python 3.12.9 is available and used
step "Installing Python 3.12.9 with uv"
#uv python install 3.12.9

step "Creating virtual environment with Python 3.12.9"
#uv venv --python 3.12.9

step "Pinning Python 3.12.9"
#uv python pin 3.12.9

step "Installing PyTorch packages"
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

step "Installing vLLM"
uv pip install vllm==0.8.5

step "Installing requirements.txt dependencies"
uv pip install -r requirements.txt 

step "Installing flash-attn"
uv pip install flash-attn==2.7.3 --no-build-isolation
