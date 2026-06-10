#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="$1"
TASK="$2"
OUTPUT_PATH="$3"
DEVICE="${4:-cuda:0}"
BATCH_SIZE="${5:-1}"


lm_eval \
  --model hf \
  --model_args "pretrained=${MODEL_PATH},dtype=float16" \
  --tasks "${TASK}" \
  --device "${DEVICE}" \
  --batch_size "${BATCH_SIZE}" \
  --limit 200 \
  --output_path "${OUTPUT_PATH}" \
  --trust_remote_code
