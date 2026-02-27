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