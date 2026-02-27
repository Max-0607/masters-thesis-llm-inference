model_name=huggyllama/llama-7B
tasks=winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai

# Run evaluation on the original model
python evaluate.py --model hf \
    --model_args pretrained=${model_name},dtype=float16 \
    --tasks ${tasks} \
    --device cuda:0 \
    --batch_size 16 \
    --output_path outputs/original;

# Run evaluation removing super weight, then removing super weight but keeping
# the induced super activation
for outlier_method in manual_scaling_SO_0.0 removeSW_restoreSA;
do
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
        --tasks $tasks \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${outlier_method};
done