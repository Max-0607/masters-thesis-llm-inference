# Reproducing the Thesis Experiments

This document describes how to reproduce the figures, tables, and evaluation results presented in the Master's thesis

> **An Empirical Study of Super Weights and Their Role in the Quantization of Large Language Models**

The commands below are intended to be executed from the repository root. Visible prose consistently uses *super weight* as two words. Technical identifiers, script names, and existing directory names such as `results/superweights/` remain unchanged.

## Result Types

The repository contains two complementary result collections:

- `results/superweights/` and `results/quantization/` contain the original single-run experiments, exploratory analyses, supplementary evaluations, and hyperparameter studies.
- `results/uncertainty/` contains the final multi-seed results reported for Tables 3.2, 3.8, and 5.1–5.5.

The final uncertainty evaluation uses five evaluation seeds:

```text
42, 43, 44, 45, 46
```

These seeds vary the evaluated sample, not the model initialization. Within a seed, compared methods use the same selected example IDs. For XCOPA-en, all 100 available examples are evaluated in every run; therefore, the selected sample and reported score are identical across seeds.

## General Setup

Create and activate the project environment, install the required dependencies, and run all commands from the repository root:

```bash
cd ~/masters-thesis-llm-inference
source venv/bin/activate
export PYTHONPATH=.
```

The experiments require access to the corresponding Hugging Face checkpoints and sufficient GPU memory. Some 7B-model evaluations were run on an H100 MIG instance with small batch sizes.

---

# Chapter 3 — Super Weight Analysis

## Figure 3.2 — Visualization of Activation Spikes

The following command records the maximum absolute input and output activations of the MLP down-projection in every transformer layer:

```bash
PYTHONPATH=. python \
scripts/superweights/3_3_activation_analysis/plot_base_input_output.py \
  --model-key mistral-7b \
  --run-name mistral-7b \
  --output_dir outputs/activation_analysis
```

Replace `mistral-7b` with another key defined in `configs/models.py` to analyze a different model. The script writes a PNG plot and a JSON file containing the layer-wise activation summaries to `outputs/activation_analysis/`.

## Table 3.1 — Top-10 Global Super Weight Candidates

The category scan aggregates activation spikes across predefined prompts and ranks candidate positions by their combined input-output activation score:

```bash
PYTHONPATH=. python \
scripts/superweights/3_3_activation_analysis/run_category_scan.py \
  --model_key olmo-1b \
  --output_dir outputs/category_scan \
  --top_k 10
```

The principal output is `outputs/category_scan/olmo-1b_top10_global.json`. Additional JSON files and plots contain category-specific activation summaries and candidates.

## Figure 3.3 — Task-Level Sensitivity to Super Weight Ablation

### Generate ablated checkpoints

`run_ablation.py` creates one checkpoint for the requested value of `--top_k`. Run it once for each ablation level required by the figure:

```bash
for K in 1 2 3 4 5 10; do
  PYTHONPATH=. python \
  scripts/superweights/3_4_superweight_ablation/run_ablation.py \
    --model_key olmo-1b \
    --candidate_json outputs/category_scan/olmo-1b_top10_global.json \
    --top_k "$K" \
    --output_dir outputs/ablated_models
done
```

The generated checkpoints are stored under `outputs/ablated_models/`.

### Evaluate the checkpoints

The original task evaluations were performed with the task-specific evaluation pipeline and, for selected larger models, the archived LLMSuperWeight notebooks under `archive/reference/LLMSuperWeight/`. The exact numerical outputs used for the analysis are retained in `results/superweights/`. Final five-seed ablation results are retained in `results/uncertainty/table_3_2/`.

### Generate the heatmap

```bash
PYTHONPATH=. python \
scripts/superweights/3_4_superweight_ablation/plot_olmo_topk_heatmap.py
```

The generated figure is written to `outputs/plots/`.

## Figure 3.4 — Category-Level Effects of Super Weight Removal

This analysis compares token probabilities before and after super weight removal:

```bash
PYTHONPATH=. python \
scripts/superweights/3_6_category_effects/plot_category_shift.py \
  --original results/stopword_probability/olmo_1b_original.json \
  --no-sw results/stopword_probability/olmo_1b_no_sw.json \
  --output-dir outputs/category_shift/olmo-1b
```

The script generates:

```text
outputs/category_shift/olmo-1b/category_level_shift.png
outputs/category_shift/olmo-1b/lowest_token_shifts_by_category.png
```

Detailed token-level results are stored in `results/category_analysis/`.

## Figure 3.5 — Super Weight Scaling Analysis

The scaling plot is generated from the existing task-evaluation JSON files in `outputs/eval/olmo_sw1_scaling/`:

```bash
PYTHONPATH=. python \
scripts/superweights/3_7_scaling/plot_olmo_sw1_scaling_heatmap.py
```

The resulting heatmap is written to `outputs/plots/olmo_sw1_scaling_delta_heatmap.png`.

## Figures 3.6–3.7 — Knowledge Redistribution

### Original model activations

```bash
PYTHONPATH=. python \
scripts/superweights/3_3_activation_analysis/plot_base_input_output.py \
  --model-key olmo-1b \
  --run-name original_olmo1b \
  --output_dir outputs/activation_analysis/redistribution_comparison
```

### Redistributed model activations

```bash
PYTHONPATH=. python \
scripts/superweights/3_3_activation_analysis/plot_base_input_output.py \
  --model-key olmo-1b \
  --model-path results/superweights/gradient_zeroing/olmo1b/models/sw_dropout_p0_25_500steps \
  --run-name sw_dropout_olmo1b \
  --output_dir outputs/activation_analysis/redistribution_comparison
```

### Activation-distribution comparison

```bash
PYTHONPATH=. python \
scripts/superweights/3_8_redistribution/plot_activation_distribution_comparison.py \
  --model-key olmo-1b \
  --redistribution-model-path results/superweights/gradient_zeroing/olmo1b/models/sw_dropout_p0_25_500steps \
  --output-dir outputs/activation_analysis/redistribution_comparison
```

The associated training and analysis scripts are located in `scripts/superweights/3_8_redistribution/`.

## Table 3.2 — Impact of Super Weight Ablation

The final table reports HellaSwag `acc_norm` over evaluation seeds 42–46 for the baseline, random-ablation, and super weight ablation conditions. The committed results are stored by model in `results/uncertainty/table_3_2/`. Original and supplementary single-run results remain in `results/superweights/`.

## Table 3.8 — Knowledge Redistribution

The redistributed checkpoints are evaluated with `scripts/superweights/3_8_redistribution/eval_redistribution_task.py`.

Example for one OLMo-1B baseline seed:

```bash
PYTHONPATH=. python \
scripts/superweights/3_8_redistribution/eval_redistribution_task.py \
  --model-key olmo1b \
  --task hellaswag \
  --split validation \
  --limit 500 \
  --eval-seed 42 \
  --max-length 2048 \
  --output-json results/uncertainty/table_3_8/olmo1b/baseline_seed42.json
```

For paired comparisons, pass the corresponding Table 3.2 baseline JSON via `--reference-json`. Add `--ablate-superweights` for the ablated condition. For redistribution, additionally pass the redistributed checkpoint via `--model-path`. Final results are stored in `results/uncertainty/table_3_8/`.

---

# Chapter 5 — Quantization

Original single-run experiments and hyperparameter studies are stored in `results/quantization/`. Final five-seed results for Tables 5.1–5.5 are stored in `results/uncertainty/`.

## Table 5.1 — Activation Bit-Width

Activation-quantized language-model perplexity is evaluated with `scripts/quantization/eval_task/eval_activation_quant.py`.

Example FP16 WikiText-2 run:

```bash
PYTHONPATH=. python \
scripts/quantization/eval_task/eval_activation_quant.py \
  --model-key llama-7b \
  --mode fp16 \
  --bits 8 \
  --dtype float16 \
  --dataset wikitext2 \
  --split validation \
  --limit 128 \
  --max-length 512 \
  --eval-seed 42 \
  --output-json results/uncertainty/table_5_1/wikitext2/fp16_seed42.json
```

Use `--mode naive` for ordinary activation quantization and `--mode super` for super weight-aware activation quantization. Use `--bits 8` for W16A8 and `--bits 4` for W16A4. For paired comparisons, the quantized runs should reuse the FP16 sample through `--reference-json`.

Final results are stored in:

```text
results/uncertainty/table_5_1/wikitext2/
results/uncertainty/table_5_1/c4/
```

## Table 5.2 — Downstream Tasks

Downstream evaluations use task-specific scripts:

```text
scripts/quantization/eval_task/eval_boolq.py
scripts/quantization/eval_task/eval_hellaswag.py
scripts/quantization/eval_task/eval_winogrande.py
scripts/quantization/eval_task/eval_xcopa.py
```

Further task scripts, including PIQA, ARC, MGSM, and MMLU variants, are retained in the same directory.

Example BoolQ run:

```bash
PYTHONPATH=. python \
scripts/quantization/eval_task/eval_boolq.py \
  --model-key olmo-1b \
  --mode fp16 \
  --bits 8 \
  --dtype float16 \
  --split validation \
  --limit 500 \
  --eval-seed 42 \
  --max-length 512 \
  --normalize-by-length \
  --output-json results/uncertainty/table_5_2/boolq/fp16_seed42.json
```

Use the same seed and evaluation settings for all methods being compared. The final results are organized by task in `results/uncertainty/table_5_2/`.

## Table 5.3 — Multilingual Evaluation

FLORES perplexity is evaluated with `scripts/quantization/eval_task/eval_flores.py`.

Example German-to-English FP16 run:

```bash
PYTHONPATH=. python \
scripts/quantization/eval_task/eval_flores.py \
  --model-key olmo-1b \
  --mode fp16 \
  --bits 4 \
  --dtype float16 \
  --src-lang de \
  --tgt-lang en \
  --limit 50 \
  --eval-seed 42 \
  --output-json results/uncertainty/table_5_3/de_to_en/fp16_seed42.json
```

For paired comparisons, use the FP16 result as `--reference-json` for the corresponding naive and super weight-aware runs. Final language-pair directories are:

```text
results/uncertainty/table_5_3/de_to_en/
results/uncertainty/table_5_3/en_to_de/
results/uncertainty/table_5_3/en_to_es/
results/uncertainty/table_5_3/en_to_fr/
results/uncertainty/table_5_3/es_to_en/
results/uncertainty/table_5_3/fr_to_en/
```

## Table 5.4 — Model-Size Comparison

This table combines activation-quantization results for OLMo-1B and OLMo-7B. The experiments use `eval_activation_quant.py` with the procedure described for Table 5.1.

Final results are stored in:

```text
results/uncertainty/table_5_4/olmo-1b/
results/uncertainty/table_5_4/olmo-7b/
```

Each model directory contains separate `wikitext2/` and `c4/` subdirectories.

## Table 5.5 — Quantization-Method Comparison

The table compares FP16, naive W4, Super Weight-Aware W4, GPTQ W4, and AWQ W4 on BoolQ, HellaSwag, WinoGrande, and XCOPA-en.

### FP16, naive, and Super Weight-Aware Quantization

Use the task-specific scripts in `scripts/quantization/eval_task/` with `--bits 4` and the appropriate `--mode`.

### GPTQ

GPTQ evaluation scripts are located in `scripts/quantization/gptq/`. They include task-specific implementations for BoolQ, HellaSwag, WinoGrande, and XCOPA-en, as well as perplexity evaluation.

### AWQ

AWQ evaluation scripts are located in `scripts/quantization/awq/`. They include task-specific implementations for BoolQ, HellaSwag, PIQA, SciQ, WinoGrande, and XCOPA-en, as well as perplexity evaluation.

Final five-seed results are stored by task in `results/uncertainty/table_5_5/`. GPTQ uses a fixed calibration seed while the evaluation seed varies. All methods within an evaluation seed use identical selected example IDs.

## Table 5.6 — Super Weight Scaling

The complete scaling study is stored in:

```text
results/quantization/olmo1b/Superweight Scaling/super_w8_scaling/
```

This hyperparameter study is part of the original experiment results and is not included in the five-seed uncertainty evaluation.

## Table 5.7 — Protected Super Weights in AWQ (SW-AWQ)

The task-specific SW-AWQ evaluation scripts are located in `scripts/quantization/protected_superweights_awq/`.

The result directory contains the hyperparameter search, selected values, and final benchmark evaluations:

```text
results/quantization/olmo1b/protected_superweights_awq/
```

This experiment is not included in the five-seed uncertainty evaluation.

---

# Repository Output Structure

| Thesis result | Primary repository location |
|---|---|
| Figure 3.2 and Figures 3.6–3.7 | `outputs/activation_analysis/` |
| Table 3.1 | `outputs/category_scan/` |
| Figures 3.3–3.5 | `outputs/plots/` |
| Original Chapter 3 evaluations | `results/superweights/` |
| Table 3.2 final results | `results/uncertainty/table_3_2/` |
| Table 3.8 final results | `results/uncertainty/table_3_8/` |
| Original Chapter 5 evaluations | `results/quantization/` |
| Table 5.1 final results | `results/uncertainty/table_5_1/` |
| Table 5.2 final results | `results/uncertainty/table_5_2/` |
| Table 5.3 final results | `results/uncertainty/table_5_3/` |
| Table 5.4 final results | `results/uncertainty/table_5_4/` |
| Table 5.5 final results | `results/uncertainty/table_5_5/` |
| Tables 5.6–5.7 | corresponding subdirectories of `results/quantization/` |

`outputs/` contains generated figures, checkpoints, and intermediate artifacts. `results/` contains numerical evaluation results. The committed JSON files preserve the configurations, metrics, selected example IDs, and per-seed outputs required to trace the reported tables.
