#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3.5-9B"
MODEL_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
VLLM_VERSION="0.29.0"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -d .venv-vllm ]]; then
  uv venv --python 3.12 .venv-vllm
fi

source .venv-vllm/bin/activate
uv pip install "vllm==${VLLM_VERSION}" --torch-backend=auto

exec vllm serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --tokenizer-revision "${MODEL_REVISION}" \
  --served-model-name "${MODEL_ID}" \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --generation-config vllm \
  --host 0.0.0.0 \
  --port 8000
