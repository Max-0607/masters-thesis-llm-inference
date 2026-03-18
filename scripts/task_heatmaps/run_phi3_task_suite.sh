#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/task_heatmaps/phi3_tasks.yaml}"
MODEL_DIR="${2:?Please provide model dir}"
OUTPUT_DIR="${3:-outputs/task_heatmaps/phi3}"
DEVICE="${4:-cuda:0}"
BATCH_SIZE="${5:-1}"

mkdir -p "${OUTPUT_DIR}"

TASKS=(
  mgsm_direct_en
  hellaswag
  sciq
  lambada_openai
)

for TASK in "${TASKS[@]}"; do
  OUTPUT_FILE="${OUTPUT_DIR}/$(basename "${MODEL_DIR}")__${TASK}.json"

  if [ -f "${OUTPUT_FILE}" ]; then
    echo "Skipping ${TASK}, file already exists: ${OUTPUT_FILE}"
    continue
  fi

  echo "Running task: ${TASK}"
  bash scripts/task_heatmaps/run_lm_eval_simple.sh \
    "${MODEL_DIR}" \
    "${TASK}" \
    "${OUTPUT_FILE}" \
    "${DEVICE}" \
    "${BATCH_SIZE}"
done