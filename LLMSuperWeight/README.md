# The Super Weight in Large Language Models

Pruning as few as a **single parameter** can destroy an LLM's ability to
generate text -- increasing perplexity by 3 orders of magnitude and reducing
zero-shot accuracy to guessing. We propose a data-free method for identifying
such parameters, termed *super weights*, using a single forward pass through
the model.

To reproduce any of the results from the paper, use the corresponding bash
scripts found in the `scripts/`. *Run these bash scripts from the root of the
repository*, such as `bash scripts/table1_superweight_importance.sh`.

```
scripts/
    table1_superweight_importance.sh
    figure3_how_to_identify_superweight.sh
    figure4_super_activation.sh
```

This repository supports the following models.

```
allenai/OLMo-1B-0724-hf
allenai/OLMo-7B-0724-hf
mistralai/Mistral-7B-v0.1
mistralai/Mistral-7B-Instruct-v0.1
microsoft/Phi-3-mini-4k-instruct
huggyllama/llama-7B # up to 30B
```

## Ablation: Importance of SW 

```
model_name=huggyllama/llama-7B

outlier_method=search_percentage_1e-6_0-32_all
for restore_and_scale_GO in 1.0 0.0
do
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},restore_and_scale_GO=${restore_and_scale_GO},dtype=float16 \
        --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/search/${outlier_method}_restore_and_scale-${restore_and_scale_GO} \

done


for outlier_method in manual_scaling_SO_0.0 manual_scaling_SO_1.0
do

python evaluate.py --model hf-outlier \
    --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
    --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
    --device cuda:0 \
    --batch_size 16 \
    --output_path outputs/${model_name}/sensitivity/${outlier_method} \

done
```


## Sensitity of SW

```
model_name=huggyllama/llama-7B

for scale in 0.0 0.2 0.5 0.8 1.0 1.5 2.0 3.0
do
    outlier_method=manual_scaling_SO_${scale}
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
        --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/sensitivity/${outlier_method} \

done
```


## Block-wise Weight Quantization
```
model_name=huggyllama/llama-7B
# Baseline: INT4, no scale-shift
for blocksize in tensor 1048576 262144 65536 16384
do
    manual_quantize=minmax_4_${blocksize}_no_0_False_False
    restore_and_scale_GO=False
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
        --tasks wikitext,winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
        --device cuda:0 \
        --batch_size 4 \
        --output_path outputs/${model_name}/groupwise/int4/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
        --trust_remote_code \

done


for blocksize in tensor 1048576 262144 65536 16384
do
    restore_and_scale_GO=1.0
    for manual_quantize in clip_4_${blocksize}_z_9_False_False clip_4_${blocksize}_bp_1e-5_False_False clip_4_${blocksize}_tp_1e-6_False_False
    do
        python evaluate.py --model hf-outlier \
            --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
            --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
            --device cuda:0 \
            --batch_size 4 \
            --output_path outputs/${model_name}/groupwise/int4/ours/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
            --trust_remote_code \

    done

done

```

