# model_name=huggyllama/llama-13B
model_name=huggyllama/llama-7B
\
outlier_method=search_percentage_5e-7_0-32_all
for restore_and_scale_GO in True False
do
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},restore_and_scale_GO=${restore_and_scale_GO} \
        --tasks wikitext \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/search/${outlier_method}_restore_and_scale-${restore_and_scale_GO} \

done
    # --model_args pretrained=huggyllama/llama-7b \
    # --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada \
