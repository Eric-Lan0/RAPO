export https_proxy=http://9.21.0.122:11113
export http_proxy=http://9.21.0.122:11113
export ftp_proxy=http://9.21.0.122:11113
# export HF_ENDPOINT=https://hf-mirror.com

# conda create -n lm-evaluation python=3.10.9
# conda activate lm-evaluation
# cd lm-evaluation-harness
# pip install -e .
# pip install -e ".[vllm]"
# pip install -e ".[math]"
# pip install -e ".[ifeval]"
# # pip install lm_eval[vllm]
# # pip install lm_eval[math]

# conda create -p /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/lm-evaluation python=3.10.9
# conda activate /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/lm-evaluation
# bash run_evaluation.sh

# Base paths
algo="rapo"  # rapo
method="const"  # const ra app_1 app_2
lamb=0.5  # lambda
i=3  # iteration index
style="online"  # online offline

my_world_size=1  # number of GPUs
model_replicas=8  # data_parallel

if [ $algo == "dpo" ]; then
    location=$algo
    prefix="${location}_"
else
    location="${algo}_${method}"
    if [ $method == "const" ]; then
        prefix="${location}_lamb${lamb}_"
    else
        prefix="${location}_"
    fi
fi
iteration_name="LLaMA3_iter${i}"
# model_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/checkpoint/${style}/${location}/${prefix}${iteration_name}"
# output_path="/apdcephfs_us/share_300814644/user/ericglan/lm-evaluation-harness/eval_results/${style}/${location}/"

model_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/sft_checkpoint/LLaMA3-SFT"
output_path="/apdcephfs_us/share_300814644/user/ericglan/lm-evaluation-harness/eval_results/sft/"


# # accelerate
# accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=${model_path} \
#     --tasks lambada_openai,arc_easy \
#     --batch_size 8 \
#     --output_path ${output_path}

task="mmlu"
# large models
lm_eval --model hf \
    --model_args pretrained=${model_path},trust_remote_code=True,parallelize=True \
    --tasks "${task}" \
    --batch_size 8 \
    --output_path "${output_path}/${task}/"


# # vllm
# lm_eval --model vllm \
#     --model_args pretrained=${model_path},trust_remote_code=True,tensor_parallel_size=${my_world_size},dtype=auto,gpu_memory_utilization=0.8,data_parallel_size=${model_replicas} \
#     --tasks math_word_problems \
#     --batch_size auto \
#     --output_path ${output_path}

# leaderboard_math_hard, math_word_problems, mathqa, ifeval,gpqa,mmlu,hellaswag,truthfulqa,gsm8k
