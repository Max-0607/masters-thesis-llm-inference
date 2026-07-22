# Super Weights and Quantization in Large Language Models

This repository contains the implementation, experimental pipeline, and evaluation results accompanying the Master's thesis

> **An Empirical Study of Super Weights and Their Role in the Quantization of Large Language Models**
>
> Maximilian Knell  
> University of Mannheim, 2026

The repository covers the experiments presented in the thesis, including super weight identification and ablation, knowledge redistribution, and super weight-aware quantization across multiple large language models.

---

## Thesis Overview

Large language models contain billions of parameters, yet recent research has shown that a very small subset of individual parameters—referred to as **super weights**—can have a disproportionate influence on model behavior.

This thesis investigates

- the identification of super weights through activation spikes;
- the effect of super weight ablation on downstream performance;
- the redistribution of super weight functionality through retraining;
- the interaction between super weights and quantization;
- extensions of Super Weight-Aware Quantization;
- and combinations of super weight protection with AWQ and GPTQ.

The repository provides the code, intermediate outputs, final evaluation results, and figures used for these analyses.

---

## Repository Structure

```text
masters-thesis-llm-inference/
├── src/                    Core implementation
├── scripts/                Experiment and evaluation scripts
├── outputs/                Intermediate outputs and generated plots
├── results/                Numerical experimental results
├── quantization/           Adapted AWQ and GPTQ implementations
├── dashboard/              Streamlit visualization dashboard
└── README.md
```

---

## Experimental Results

The experimental outputs are organized according to the corresponding chapters and tables of the thesis.

The repository distinguishes between two types of numerical results:

- `results/superweights/` and `results/quantization/` contain the original single-run experiments, exploratory analyses, supplementary evaluations, and hyperparameter studies.
- `results/uncertainty/` contains the final multi-seed results reported for Tables 3.2, 3.8, and 5.1–5.5.

---

## Chapter 3 — Super Weight Analysis

### Activation Analysis

```text
outputs/activation_analysis/
```

Contains activation analyses for the evaluated language models, including

- maximum input activations;
- maximum output activations;
- activation spike plots;
- and activation plots before and after knowledge redistribution.

These outputs correspond to

- Figure 3.2;
- Figures 3.6–3.7;
- and Appendix A.4.

---

### Global Super Weight Candidate Scan

```text
outputs/category-scan/
```

Contains the activation-based search for super weight candidates, including

- ranked super weight candidates;
- activation scores;
- layer indices;
- and parameter coordinates.

These outputs correspond to Table 3.1.

---

### Generated Figures

```text
outputs/plots/
```

Contains generated figures used throughout the thesis, including

- Figure 3.3 — Task-Level Performance Drop under Top-k Super Weight Ablation;
- Figure 3.4 — Category-Level Shift in Token Probabilities;
- and Figure 3.5 — Task-Specific Performance under Super Weight Scaling.

---

### Super Weight Evaluation

```text
results/superweights/
```

Contains the original and supplementary evaluation results for

- super weight ablation;
- random weight ablation;
- knowledge redistribution;
- restored models;
- and downstream benchmark evaluations.

The final five-seed results reported in Tables 3.2 and 3.8 are stored separately in

```text
results/uncertainty/table_3_2/
results/uncertainty/table_3_8/
```

---

### Category Analysis

```text
results/category_analysis/
```

Contains

- token-probability shifts;
- linguistic-category statistics;
- and detailed token-level analyses.

These outputs correspond to

- Figure 3.4;
- and Appendix A.5.

---

## Chapter 5 — Quantization

The original quantization experiments, supplementary evaluations, and hyperparameter studies are located in

```text
results/quantization/
```

For Tables 5.1–5.5, the final results reported in the thesis are based on five evaluation seeds and are stored in the corresponding subdirectories of

```text
results/uncertainty/
```

The original experiment directories remain available for traceability and supplementary analysis.

---

### Activation Bit-Width

Original experiment outputs:

```text
results/quantization/llama7b/activation_bit-width/
```

Final five-seed results reported in Table 5.1:

```text
results/uncertainty/table_5_1/
```

Contains perplexity evaluations for

- FP16;
- W16A8;
- and W16A4.

---

### Downstream Task Evaluation

Original experiment outputs:

```text
results/quantization/llama7b/tasks/
```

Final five-seed results reported in Table 5.2:

```text
results/uncertainty/table_5_2/
```

Contains downstream evaluations on tasks such as

- BoolQ;
- HellaSwag;
- PIQA;
- WinoGrande;
- XCOPA;
- and SciQ.

---

### Multilingual Evaluation

Original experiment outputs:

```text
results/quantization/llama7b/language/flores/
```

Final five-seed results reported in Table 5.3:

```text
results/uncertainty/table_5_3/
```

Contains multilingual perplexity evaluations based on FLORES language pairs.

---

### Model-Size Comparison

Original experiment outputs:

```text
results/quantization/olmo1b/Activation Bit-Width/
```

Final five-seed results reported in Table 5.4:

```text
results/uncertainty/table_5_4/
```

Contains activation-quantization experiments comparing models of different sizes.

---

### Quantization-Method Comparison

Original experiment outputs:

```text
results/quantization/olmo1b/Quantization Method/
```

Final five-seed results reported in Table 5.5:

```text
results/uncertainty/table_5_5/
```

The evaluated methods include

- naive round-to-nearest quantization;
- Super Weight-Aware Quantization;
- GPTQ;
- and AWQ.

---

### Super Weight Scaling

```text
results/quantization/olmo1b/Superweight Scaling/super_w8_scaling/
```

Contains the complete super weight scaling study reported in Table 5.6.

This experiment is not part of the five-seed uncertainty evaluation.

---

### Protected Super Weights in AWQ

```text
results/quantization/olmo1b/protected_superweights_awq/
```

Contains the SW-AWQ experiments and associated hyperparameter sweeps reported in Table 5.7.

This experiment is not part of the five-seed uncertainty evaluation.

---

## Final Results and Uncertainty Evaluation

The final results reported for Tables 3.2, 3.8, and 5.1–5.5 are stored in

```text
results/uncertainty/
```

These results are based on five evaluation seeds:

```text
42, 43, 44, 45, 46
```

Each JSON file contains the result for one model, method, task, and evaluation seed. The thesis reports the mean and standard deviation across the corresponding runs.

The evaluation seeds control the selection of evaluation examples. They do not represent different model initializations. The same pretrained or quantized model configuration is retained across evaluation seeds.

The directory is organized according to the corresponding thesis tables:

```text
results/uncertainty/
├── table_3_2/    Super weight and random weight ablation
├── table_3_8/    Knowledge redistribution
├── table_5_1/    Activation bit-width comparison
├── table_5_2/    Downstream task evaluation
├── table_5_3/    Multilingual evaluation
├── table_5_4/    Model-size comparison
└── table_5_5/    Quantization-method comparison
```

For XCOPA-en, all 100 available evaluation examples are used in every run. Consequently, the evaluated sample is identical across seeds, resulting in a standard deviation of zero.

Tables 5.6 and 5.7 are not part of the five-seed uncertainty evaluation. Their results remain in the corresponding subdirectories of `results/quantization/`.

---

## Implemented Methods

### Super Weight Analysis

- Activation Spike Analysis
- Super Weight Identification
- Super Weight Ablation
- Random Weight Ablation
- Task Sensitivity Analysis
- Category-Level Analysis
- Super Weight Scaling

### Knowledge Redistribution

- Super Weight Dropout
- Gradient Zeroing
- Redistribution Training
- Activation Concentration Analysis
- Post-Training Super Weight Ablation

### Quantization

- Naive Round-to-Nearest Quantization
- Super Weight-Aware Quantization
- Activation Quantization
- Weight Quantization
- GPTQ
- AWQ
- Super Weight Scaling
- SW-AWQ

---

## Supported Models

- OLMo-1B
- OLMo-7B
- LLaMA-7B
- Mistral-7B
- Phi-3 Mini

---

## Evaluation Benchmarks

### Reasoning and Question Answering

- HellaSwag
- BoolQ
- PIQA
- WinoGrande
- XCOPA
- MGSM
- SciQ

### Language Modeling

- WikiText-2
- C4

### Multilingual Evaluation

- FLORES

---

## Dashboard

The repository includes a Streamlit dashboard for exploring selected experimental results and visualizations.

Run the dashboard from the repository root with

```bash
streamlit run dashboard/app.py
```

---

## Acknowledgements

This repository builds upon the following open-source projects:

- [LLMSuperWeight](https://github.com/mengxiayu/LLMSuperWeight)
- [AWQ](https://github.com/mit-han-lab/llm-awq)
- [GPTQ](https://github.com/IST-DASLab/gptq)

The original implementations were adapted and extended for the experiments conducted in this thesis.

---

## Citation

If you use this repository, please cite

```text
Maximilian Knell.
An Empirical Study of Super Weights and Their Role in the Quantization
of Large Language Models.
Master's Thesis, University of Mannheim, 2026.
```