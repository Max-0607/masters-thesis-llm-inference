#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/baseline_fp16

bash scripts/evaluation/run_lm_eval.sh \
  mistralai/Mistral-7B-v0.1 \
  mgsm_direct_en \
  outputs/baseline_fp16/mistral7b_mgsm_direct_en_fp16.json \
  cuda:0 \
  1

bash scripts/evaluation/run_lm_eval.sh \
  mistralai/Mistral-7B-v0.1 \
  mgsm_direct_es \
  outputs/baseline_fp16/mistral7b_mgsm_direct_es_fp16.json \
  cuda:0 \
  1

bash scripts/evaluation/run_lm_eval.sh \
  mistralai/Mistral-7B-v0.1 \
  mgsm_direct_fr \
  outputs/baseline_fp16/mistral7b_mgsm_direct_fr_fp16.json \
  cuda:0 \
  1

bash scripts/evaluation/run_lm_eval.sh \
  mistralai/Mistral-7B-v0.1 \
  mgsm_direct_ja \
  outputs/baseline_fp16/mistral7b_mgsm_direct_ja_fp16.json \
  cuda:0 \
  1
