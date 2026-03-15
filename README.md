# Masters Thesis – Quantization and Superweights in Large Language Models

This repository contains the code and experiments for my master's thesis on **quantization and superweights in large language models (LLMs)**.

## Research Goal

The goal of this project is to analyze:

- how **quantization affects multilingual LLM performance**
- whether some **languages are more sensitive to quantization**
- whether **specific tasks degrade more strongly**
- whether **superweights play a key role in model robustness**
- whether **superweight-aware quantization can reduce performance loss**

---

# Models

The experiments focus on the following models:

- **Mistral-7B**
- **Phi-3-mini-4k-instruct**
- **OLMo-7B**

These models were chosen because they provide different architectures and levels of superweight concentration.

---

# Benchmarks

Initial benchmarks include:

- **mMMLU** – multilingual knowledge benchmark
- **MGSM** – multilingual math reasoning
- **XCOPA** – multilingual causal reasoning
- **FLORES** – translation benchmark

---

# Quantization Methods

The following quantization methods are compared:

- FP16 / BF16 baseline
- RTN (Round-To-Nearest)
- GPTQ
- AWQ
- Superweight-aware quantization (proposed method)

---

# Evaluation Framework

All evaluations are performed using:

**lm-evaluation-harness**

This allows consistent evaluation across multiple tasks and languages.

---

# Repository Structure
masters-thesis-llm-inference
│
├── external/ # external repositories (GPTQ, AWQ, Superweights)
├── configs/ # model, task, and experiment configs
├── scripts/ # reproducible experiment scripts
├── src/ # thesis-specific Python code
├── notebooks/ # exploratory analysis
├── outputs/ # raw experiment outputs
├── results/ # aggregated results and figures


---

# Planned Experiments

1. Baseline multilingual performance
2. Standard quantization comparison
3. Language sensitivity analysis
4. Superweight ablation experiments
5. Superweight-aware quantization
6. Activation analysis across languages

---

# Author

Master's thesis project  
University of Mannheim
