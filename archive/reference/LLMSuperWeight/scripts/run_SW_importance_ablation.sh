# model_name=huggyllama/llama-13B
# model_name=huggyllama/llama-7B
# model_name=mistralai/Mistral-7B-v0.1

# model_name=microsoft/Phi-3-mini-4k-instruct

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