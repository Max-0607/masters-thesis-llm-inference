# Superweights and Quantization in Large Language Models

This repository contains the complete implementation, experimental pipeline, and evaluation results accompanying the Master's Thesis

> **An Empirical Study of Superweights and Their Role in Quantization of Large Language Models**  
> Maximilian Knell  
> University of Mannheim, 2026

The repository reproduces all experiments presented in the thesis, including superweight analysis, knowledge redistribution, and superweight-aware quantization across multiple large language models.

---

# Thesis Overview

Large Language Models contain billions of parameters, yet recent work has shown that only a very small subset—called **superweights**—has a disproportionate influence on model behavior.

This thesis investigates

- identification of superweights via activation spikes
- impact of superweight ablation on downstream performance
- redistribution of superweight functionality through retraining
- interaction between superweights and quantization
- extensions of Superweight-Aware Quantization
- combinations with AWQ and GPTQ

The implementation reproduces all experiments and figures presented in the thesis.

---

# Repository Structure

```
masters-thesis-llm-inference/

├── src/                    Core implementation
├── scripts/                Scripts for reproducing experiments
├── outputs/                Intermediate outputs and generated plots
├── results/                Final experimental results
├── quantization/           Adapted AWQ and GPTQ implementations
├── dashboard/              Streamlit visualization dashboard
└── README.md
```

---

# Experimental Outputs

The repository is organized according to the chapters of the thesis.

## Chapter 3 — Superweight Analysis

### Activation Analysis

```
outputs/activation_analysis/
```

Contains the activation analysis for all evaluated language models.

Includes

- maximum input activations
- maximum output activations
- activation spike plots
- redistribution activation plots

Corresponds to

- Figure 3.2
- Figures 3.6–3.7
- Appendix A.4

---

### Global Superweight Candidate Scan

```
outputs/category-scan/
```

Contains the activation-based search for candidate superweights.

Includes

- top-ranked superweight candidates
- activation scores
- layer indices
- coordinates

Corresponds to

- Table 3.1

---

### Generated Figures

```
outputs/plots/
```

Contains all generated figures used throughout the thesis.

Including

- Figure 3.3 — Task-Level Performance Drop under Top-k Superweight Ablation
- Figure 3.4 — Category-Level Shift in Token Probabilities
- Figure 3.5 — Task-Specific Performance under Superweight Scaling

---

### Superweight Evaluation

```
results/superweights/
```

Contains all numerical evaluation results for

- superweight ablation
- random ablation
- redistribution
- restored models
- benchmark evaluations

Corresponds to

- Table 3.2
- Table 3.3

---

### Category Analysis

```
results/category_analysis/
```

Contains

- token probability shifts
- linguistic category statistics
- detailed token-level analysis

Corresponds to

- Figure 3.4
- Appendix A.5

---

# Chapter 5 — Quantization

All quantization experiments are located in

```
results/quantization/
```

Each model contains identical experiment categories.

---

## Activation Bit-Width

Example

```
results/quantization/llama7b/activation_bit-width/
```

Contains

- FP16
- W16A8
- W16A4

perplexity results.

Corresponds to

- Table 5.1

---

## Downstream Tasks

Example

```
results/quantization/llama7b/tasks/
```

Contains benchmark evaluations on

- BoolQ
- HellaSwag
- PIQA
- WinoGrande
- XCOPA
- SciQ

Corresponds to

- Table 5.2
- Table 5.5

---

## Multilingual Evaluation

```
results/quantization/llama7b/language/flores/
```

Contains multilingual FLORES translation experiments.

Corresponds to

- Table 5.3

---

## Model Size Comparison

```
results/quantization/olmo1b/Activation Bit-Width/
```

Contains experiments comparing activation quantization across model sizes.

Corresponds to

- Table 5.4

---

## Quantization Method Comparison

```
results/quantization/olmo1b/Quantization Method/
```

Comparison of

- Naive Quantization
- Superweight Quantization
- GPTQ
- AWQ

Corresponds to

- Table 5.5

---

## Superweight Scaling

```
results/quantization/olmo1b/Superweight Scaling/super_w8_scaling/
```

Contains the complete scaling study.

Corresponds to

- Table 5.6

---

## Protected Superweights in AWQ

```
results/quantization/olmo1b/protected_superweights_awq/
```

Contains all SW-AWQ experiments and hyperparameter sweeps.

Corresponds to

- Table 5.7

---

# Implemented Methods

## Superweight Analysis

- Activation Spike Analysis
- Superweight Identification
- Superweight Ablation
- Task Sensitivity Analysis
- Category-Level Analysis
- Superweight Scaling

## Knowledge Redistribution

- Superweight Dropout
- Gradient Zeroing
- Redistribution Training
- Activation Concentration Analysis

## Quantization

- Naive RTN
- Superweight-Aware Quantization
- Activation Quantization
- Weight Quantization
- GPTQ
- AWQ
- Superweight Scaling
- SW-AWQ

---

# Supported Models

- OLMo-1B
- OLMo-7B
- LLaMA-7B
- Mistral-7B
- Phi-3 Mini

---

# Evaluation Benchmarks

Reasoning

- HellaSwag
- BoolQ
- PIQA
- WinoGrande
- XCOPA
- MGSM
- SciQ

Language Modeling

- WikiText-2
- C4

Translation

- FLORES

---

# Acknowledgements

This repository builds upon the following open-source projects:

- LLMSuperWeight
- AWQ
- GPTQ

The original implementations were adapted and extended for the experiments conducted in this thesis.

---

# Citation

If you use this repository, please cite

Maximilian Knell.
*An Empirical Study of Superweights and Their Role in Quantization of Large Language Models.*
Master's Thesis, University of Mannheim, 2026.