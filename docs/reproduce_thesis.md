# Reproducing the Thesis Experiments

This document describes how to reproduce the figures, tables, and evaluation results presented in the Master's thesis

> **An Empirical Study of Superweights and Their Role in Quantization of Large Language Models**

The documentation is organized according to the chapters of the thesis and provides the scripts, required inputs, and output locations for each experiment.

---

# Chapter 3 – Superweight Analysis

## Figure 3.2 – Visualization of Activation Spikes

### Purpose

Visualizes the maximum input and output activations of each transformer layer to identify activation spikes and potential superweights.

### Script

```bash
PYTHONPATH=. python scripts/figures/plot_olmo_base_input_output.py \
    --model-key mistral-7b \
    --output_dir outputs/activation_analysis
```

### Input

- Model checkpoint

### Output

```
outputs/activation_analysis/
```

Contains

- Maximum input activation plots
- Maximum output activation plots
- Layer-wise activation statistics

---

## Table 3.1 – Top-10 Global Superweight Candidates

### Purpose

Identifies the globally strongest activation spikes across the model.

### Script

```bash
python scripts/evaluation/run_category_scan.py \
    --model_key olmo-1b \
    --output_dir outputs/category_scan \
    --top_k 10
```

### Output

```
outputs/category_scan/
```

Contains

- Top-10 superweight candidates
- Layer indices
- Row/column coordinates
- Activation scores

---

## Figure 3.3 – Task-Level Sensitivity to Superweight Ablation

### Step 1 – Generate candidate superweights

```bash
PYTHONPATH=. python scripts/evaluation/run_ablation.py \
    --model_key olmo-1b \
    --candidate_json outputs/activation_analysis/olmo-1b/candidates.json \
    --top_k 1
```

---

### Step 2 – Generate Top-k ablated models

```
scripts/evaluation/run_ablation.py
```

Creates

- Top-1
- Top-2
- Top-3
- Top-4
- Top-5
- Top-10

ablated models.

---

### Step 3 – Evaluate models

For smaller models:

```bash
python -m lm_eval ...
```

Example

```bash
python -m lm_eval \
  --model hf \
  --model_args pretrained=huggyllama/llama-7b,dtype=float16 \
  --device cuda:0 \
  --tasks hellaswag \
  --num_fewshot 0 \
  --limit 300 \
  --batch_size 1 \
  --output_path outputs/ablation_table/llama-7b_baseline_lmeval
```

For LLaMA-7B and Mistral-7B the original LLMSuperWeight notebooks were used because `lm_eval` frequently crashed for these larger models.

---

### Step 4 – Generate heatmap

```bash
python scripts/figures/plot_olmo_topk_heatmap.py
```

### Output

```
outputs/plots/
```

---

## Figure 3.4 – Category-Level Effects of Superweight Removal

### Purpose

Analyzes how removing superweights changes token probabilities across linguistic categories.

### Script

```bash
PYTHONPATH=. python scripts/figures/plot_category_shift.py \
    --original results/stopword_probability/olmo_1b_original.json \
    --no-sw results/stopword_probability/olmo_1b_no_sw.json \
    --output-dir outputs/category_shift/olmo-1b \
    --title-prefix "OLMo-1B"
```

### Input

```
results/stopword_probability/
```

### Output

```
outputs/category_shift/
```

---

## Figure 3.5 – Superweight Scaling Analysis

### Input

```
outputs/eval/olmo_sw1_scaling/
```

Example

```
baseline_hellaswag.json
```

### Plot

```bash
PYTHONPATH=. python scripts/figures/plot_olmo_sw1_scaling_heatmap.py
```

### Output

```
outputs/plots/
```

Generated figure

```
olmo_sw1_scaling_delta_heatmap.png
```

---

## Figures 3.6–3.7 – Redistribution Analysis

### Original model

```bash
PYTHONPATH=. python scripts/figures/plot_base_input_output.py \
    --model-key olmo-1b \
    --run-name original_olmo1b \
    --output_dir outputs/activation_analysis/redistribution_comparison
```

---

### Redistribution model

```bash
PYTHONPATH=. python scripts/figures/plot_base_input_output.py \
    --model-key olmo-1b \
    --model-path results/superweights/gradient_zeroing/olmo1b/models/sw_dropout_p0_25_500steps \
    --run-name sw_dropout_olmo1b \
    --output_dir outputs/activation_analysis/redistribution_comparison
```

---

### Activation distribution comparison

```bash
PYTHONPATH=. python scripts/figures/plot_activation_distribution_comparison.py \
    --model-key olmo-1b \
    --redistribution-model-path results/superweights/gradient_zeroing/olmo1b/models/sw_dropout_p0_25_500steps \
    --output-dir outputs/activation_analysis/redistribution_comparison
```

### Output

```
outputs/activation_analysis/redistribution_comparison/
```

---

## Table 3.2 – Impact of Superweight Ablation

Evaluation performed using

```bash
python -m lm_eval ...
```

Results are stored in

```
results/superweights/
```

---

## Table 3.3 – Redistribution Performance

Evaluation of redistributed models.

Results are stored in

```
results/superweights/
```

---

# Chapter 5 – Quantization

All quantization experiments are located in

```
results/quantization/
```

---

## Table 5.1 – Activation Quantization

### Evaluation

```bash
python scripts/quantization/eval_task/eval_activation_quant.py
```

### Output

```
results/quantization/<model>/activation_bit-width/
```

Available for

- OLMo-1B
- OLMo-7B
- LLaMA-7B
- Mistral-7B

---

## Table 5.2 – Downstream Task Evaluation

### Evaluation

```bash
python scripts/quantization/eval_task/eval_task_quant.py
```

### Output

```
results/quantization/<model>/tasks/
```

Includes

- BoolQ
- HellaSwag
- PIQA
- WinoGrande
- XCOPA
- SciQ

---

## Table 5.3 – Multilingual Evaluation

### Evaluation

```bash
python scripts/quantization/eval_task/eval_language_quant.py
```

### Output

```
results/quantization/<model>/language/flores/
```

Includes

- FLORES translation
- Multilingual perplexity

---

## Table 5.4 – Model Size Comparison

Generated by combining the activation quantization results from different model sizes.

Input

```
results/quantization/olmo1b/activation_bit-width/
results/quantization/olmo7b/activation_bit-width/
```

---

## Table 5.5 – Quantization Method Comparison

### Naive / Superweight Quantization

```bash
python scripts/quantization/eval_task/eval_weight_quant.py
```

### GPTQ

```
quantization/gptq/
```

### AWQ

```
quantization/awq/
```

### Output

```
results/quantization/olmo1b/Quantization Method/
```

---

## Table 5.6 – Superweight Scaling

### Hyperparameter Sweep

```bash
python scripts/quantization/eval_task/eval_super_w8_scaling.py
```

### Plot

```bash
python scripts/figures/plot_super_w8_scaling.py
```

### Output

```
results/quantization/olmo1b/Superweight Scaling/super_w8_scaling/
```

---

## Table 5.7 – Protected Superweights in AWQ (SW-AWQ)

### Hyperparameter Search

```bash
python scripts/quantization/eval_task/eval_sw_awq.py
```

### Output

```
results/quantization/olmo1b/protected_superweights_awq/
```

Contains

- Complete hyperparameter search
- Best α₀
- Best λ
- Final benchmark results

---

# Repository Output Structure

The repository follows the organization of the thesis.

| Thesis Chapter | Repository |
|----------------|-----------|
| Chapter 3 – Superweight Analysis | `outputs/activation_analysis/`, `outputs/category_scan/`, `outputs/plots/`, `results/superweights/`, `results/category_analysis/` |
| Chapter 5 – Quantization | `results/quantization/` |

The `outputs/` directory contains generated figures and intermediate files, whereas `results/` stores the numerical evaluation results used throughout the thesis.
