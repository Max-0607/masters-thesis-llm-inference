# Superweights and Quantization in Large Language Models

This repository contains the code, experiments, and evaluation pipeline developed for my Master's Thesis at the University of Mannheim.

## Thesis Overview

The thesis investigates the role of superweights in large language models and their relationship to model compression through quantization.

Main research questions include:

- How can superweights be identified?
- How important are superweights for model performance?
- Can superweight functionality be redistributed?
- How do superweights interact with quantization?
- Can superweight-aware quantization improve existing methods such as AWQ?

## Repository Structure

```
masters-thesis-llm-inference/
│
├── src/
│   ├── quantization/
│   ├── hooks/
│   ├── models/
│   └── utils/
│
├── scripts/
│   ├── superweights/
│   ├── quantization/
│   └── evaluation/
│
├── outputs/
│   ├── plots/
│   ├── activation_analysis/
│   └── ablated_models/
│
├── results/
│   ├── superweights/
│   ├── quantization/
│   └── multilingual/
│
├── quantization/
│   ├── awq/
│   └── gptq/
│
└── README.md
```

## Implemented Experiments

### Superweight Analysis

- Activation spike analysis
- Superweight identification
- Superweight ablation
- Task sensitivity analysis
- Category-level token analysis
- Superweight scaling experiments

### Knowledge Redistribution

- Superweight dropout
- Gradient-zeroing based retraining
- Redistribution analysis
- Activation concentration analysis

### Quantization

- Naive RTN quantization
- Superweight-aware quantization
- Activation quantization
- Weight quantization
- GPTQ evaluation
- AWQ evaluation
- Superweight-aware AWQ extensions

### Evaluation

Benchmarks include:

- HellaSwag
- BoolQ
- PIQA
- WinoGrande
- SciQ
- XCOPA
- MGSM
- FLORES
- WikiText-2
- C4

Models include:

- OLMo-1B
- OLMo-7B
- LLaMA-7B
- Mistral-7B
- Phi-3 Mini

## Acknowledgements

This repository builds upon several open-source projects:

### LLMSuperWeight

Yu et al. (2025)
"The Super Weight in Large Language Models"

Repository:
https://github.com/microsoft/LLMSuperWeight

### AWQ

Lin et al. (2024)
"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

Repository:
https://github.com/mit-han-lab/llm-awq

### GPTQ

Frantar et al. (2023)
"GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

Repository:
https://github.com/IST-DASLab/gptq

The original repositories were adapted and extended for the experiments conducted in this thesis.
