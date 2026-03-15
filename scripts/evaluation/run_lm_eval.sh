#!/usr/bin/env bash

MODEL_PATH="$1"
TASKS="$2"
OUTPUT_PATH="$3"
DEVICE="${4:-cuda:0}"
BATCH_SIZE="${5:-1}"

lm_eval \
  --model hf \
  --model_args pretrained=${MODEL_PATH},dtype=float16 \
  --tasks ${TASKS} \
  --device ${DEVICE} \
  --batch_size ${BATCH_SIZE} \
  --output_path ${OUTPUT_PATH}
