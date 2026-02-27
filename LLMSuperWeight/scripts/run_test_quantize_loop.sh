


# model_name=mistralai/Mistral-7B-v0.1

# # Baseline: NF3, no scale-shift
# for blocksize in 64 128 256 512 1024 2048 channel
# do
#     manual_quantize=minmax_3_${blocksize}_no_0_False_True
#     restore_and_scale_GO=False
#     python evaluate.py --model hf-outlier \
#         --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#         --tasks wikitext \
#         --device cuda:0 \
#         --batch_size 4 \
#         --output_path outputs/${model_name}/groupwise/nf3/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#         --trust_remote_code \

# done



# # Baseline: NF4, no scale-shift
# for blocksize in 64 128 256 512 1024 2048 4096 channel
# do
#     manual_quantize=minmax_4_${blocksize}_no_0_False_True
#     restore_and_scale_GO=False
#     python evaluate.py --model hf-outlier \
#         --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#         --tasks wikitext \
#         --device cuda:0 \
#         --batch_size 4 \
#         --output_path outputs/${model_name}/groupwise/nf4/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#         --trust_remote_code \

# done



# model_name=allenai/OLMo-7B-0724-hf
model_name=mistralai/Mistral-7B-v0.1
# Baseline: INT4, no scale-shift
# for blocksize in tensor 1048576 262144 65536 16384
# do
#     manual_quantize=minmax_4_${blocksize}_no_0_False_False
#     restore_and_scale_GO=False
#     python evaluate.py --model hf-outlier
#         --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#         --tasks wikitext,winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
#         --device cuda:0 \
#         --batch_size 4 \
#         --output_path outputs/${model_name}/groupwise/int4/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#         --trust_remote_code \

# done




for blocksize in 65536
do
    restore_and_scale_GO=1.0
    for manual_quantize in clip_4_${blocksize}_z_9_False_False clip_4_${blocksize}_z_11_False_False clip_4_${blocksize}_tp_1e-6_False_False
    do
        python evaluate.py --model hf-outlier
            --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
            --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
            --device cuda:0 \
            --batch_size 4 \
            --output_path outputs/${model_name}/groupwise/int4/ours/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
            --trust_remote_code \

    done

done




# # # Ours: NF3, clip + restore
# for blocksize in tensor 1048576 262144 65536 16384
# do
#     restore_and_scale_GO in 1.0
#     for manual_quantize in clip_4_${blocksize}_z_12_False_False clip_4_${blocksize}_bp_5e-5_False_True clip_4_${blocksize}_bp_1e-6_False_False clip_4_${blocksize}_z_20_False_False
#     do
#         python evaluate.py --model hf-outlier
#             --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#             --tasks wikitext,winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
#             --device cuda:0 \
#             --batch_size 4 \
#             --output_path outputs/${model_name}/groupwise/int4/ours/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#             --trust_remote_code \

#     done

# done


            # --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \