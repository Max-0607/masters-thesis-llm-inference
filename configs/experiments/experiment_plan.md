# Experiment Plan

## Models
- Mistral-7B
- Phi-3-mini-4k-instruct
- OLMo-7B

## Languages
- en
- es
- fr
- ja
- ko

## Tasks
- mMMLU
- MGSM
- XCOPA
- FLORES

## Methods
- fp16
- rtn
- gptq
- awq
- superweight_aware

## Core Experiments
1. FP16 baseline multilingual evaluation
2. Standard quantization comparison
3. Superweight removal by language
4. Superweight removal by task
5. Superweight-aware quantization
6. Activation analysis across languages
