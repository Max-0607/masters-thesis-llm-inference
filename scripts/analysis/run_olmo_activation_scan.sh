#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:-allenai/OLMo-1B-0724-hf}"
OUTDIR="${2:-results/activation_scan/olmo1b}"
MAX_LEN="${3:-64}"

mkdir -p "${OUTDIR}"

CATEGORIES=(
  reasoning
  math
  causal
  knowledge
  coding
)

for CAT in "${CATEGORIES[@]}"; do
  echo "Running category: ${CAT}"
  python scripts/analysis/activation_scan_olmo.py \
    --model-id "${MODEL_ID}" \
    --category "${CAT}" \
    --max-len "${MAX_LEN}" \
    --output-json "${OUTDIR}/${CAT}.json"
done
