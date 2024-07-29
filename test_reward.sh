export https_proxy=http://9.21.0.122:11113
export http_proxy=http://9.21.0.122:11113
export ftp_proxy=http://9.21.0.122:11113

source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

initial_model="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/sft_checkpoint/LLaMA3-SFT"
base_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/checkpoint"
reward_model_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/reward_model/ArmoRM-Llama3-8B-v0.1"

mkdir $base_path
iteration_prefix="data_"

run_iteration() {
    local iteration=$1
    local model_path=$2
    local model_output_path=$3
    local jsonl_input=$4
    local json_output=$5
    local reward_data_path=$6
    local my_world_size=$7
    local reward_model_path=$8
    local json_output_wo=$9


    conda activate vllm
    
    accelerate launch annotate_data/get_rewards.py --dataset_name_or_path $json_output --output_dir $reward_data_path --reward_name_or_path $reward_model_path --K 4
}


my_world_size=8 # how many gpu you use

# Main loop for iterations
i=1

iteration_name="LLaMA3_iter${i}"
# /apdcephfs_us/share_300814644/user/ericglan/data/iterative-prompt-v1-iter1-20K
jsonl_input="/apdcephfs_us/share_300814644/user/ericglan/data/iterative-prompt-v1-iter${i}-20K"
json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}.json"
json_output_wo="${base_path}/${iteration_prefix}${i}_${iteration_name}"
reward_data_path="${base_path}/${iteration_prefix}${i}_${iteration_name}_reward.json"

# Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
if [ $i -eq 1 ]; then
    model_path=$initial_model
else
    previous_iteration=$((i-1))
    model_path="${base_path}/LLaMA3_iter${previous_iteration}"
fi

model_output_path="${base_path}/${iteration_name}"

run_iteration $iteration_name $model_path $model_output_path $jsonl_input $json_output $reward_data_path $my_world_size $reward_model_path $json_output_wo
