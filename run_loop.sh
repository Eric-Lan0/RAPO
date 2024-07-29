export https_proxy=http://9.21.0.122:11113
export http_proxy=http://9.21.0.122:11113
export ftp_proxy=http://9.21.0.122:11113

source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

# Base paths and settings
algo="rapo"  # rapo
method="const"  # const ra app_1 app_2
lamb=0.5  # lambda

my_world_size=8  # how many gpu you use
num_of_responses=4  # per prompt


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

#"meta-llama/Meta-Llama-3-8B-Instruct"
initial_model="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/sft_checkpoint/LLaMA3-SFT"
base_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/checkpoint/online/${location}"
reward_model_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/reward_model/ArmoRM-Llama3-8B-v0.1"

mkdir $base_path


# Function to run a set of operations for a model iteration
# run_iteration() {
#     local iteration=$1
#     local model_path=$2
#     local jsonl_input=$3
#     local json_output=$4
#     local model_output=$5

#     conda activate vllm
#     bash generation/run_4gpu.sh $model_path
#     sleep 60
#     python generation/gen_hf.py --ports 8000 8001 8002 8003 --eos_ids 128009 --tokenizer $initial_model --dataset_name_or_path $jsonl_input --output_dir $json_output --K 8 --temperature 1.0
#     pkill -f "python -m vllm.entrypoints.api_server"
#     accelerate launch annotate_data/get_rewards.py --dataset_name_or_path $json_output --output_dir $model_output --reward_name_or_path $reward_model_path
#     conda activate rlhflow
#     accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_dpo.py --run_name $iteration --output_dir $iteration --model_name_or_path $model_path --ref_model $initial_model --learning_rate 5e-7 --max_steps 1200 --choose_type max_min --train_dir $model_output --eval_dir $model_output --loss_type sigmoid --lr_scheduler_type cosine
# }

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
    local i=$10

    if [ $i != 1 ]; then
        # conda activate vllm
        conda activate /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/vllm

        CUDA_VISIBLE_DEVICES=0 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=1 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=2 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=3 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=4 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 4 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=5 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 5 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=6 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 6 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=7 python ./generation/get_hf2.py --model_name_or_path ${model_path} --dataset_name_or_path ${jsonl_input} --output_dir ${json_output_wo} --K ${num_of_responses} --temperature 1.0 --local_index 7 --my_world_size ${my_world_size} --eos_ids 128009 &

        wait
        python ./generation/merge_data.py --base_path ${json_output_wo} --output_dir ${json_output} --num_datasets ${my_world_size}
        
        accelerate launch annotate_data/get_rewards.py --dataset_name_or_path $json_output --output_dir $reward_data_path --reward_name_or_path $reward_model_path --K ${num_of_responses}
    fi

    # conda activate rlhflow
    conda activate /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/rlhflow
    accelerate launch --config_file ./configs/zero3.yaml ${algo}_iteration/run_${algo}.py --run_name $iteration --output_dir $model_output_path --model_name_or_path $model_path --ref_model $initial_model --learning_rate 5e-7 --max_steps 1200 --choose_type max_min --train_dir $reward_data_path --eval_dir $reward_data_path --loss_type sigmoid --lr_scheduler_type cosine
}

# Main loop for iterations
for i in {1..3}
do
    iteration_name="LLaMA3_iter${i}"
    # /apdcephfs_us/share_300814644/user/ericglan/data/iterative-prompt-v1-iter1-20K
    jsonl_input="/apdcephfs_us/share_300814644/user/ericglan/data/iterative-prompt-v1-iter${i}-20K"
    json_output="${base_path}/${prefix}${i}_${iteration_name}.json"
    json_output_wo="${base_path}/${prefix}${i}_${iteration_name}"
    reward_data_path="${base_path}/${prefix}${i}_${iteration_name}_reward.json"
    
    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
    else
        previous_iteration=$((i-1))
        model_path="${base_path}/${prefix}LLaMA3_iter${previous_iteration}"
    fi

    model_output_path="${base_path}/${prefix}${iteration_name}"

    run_iteration $iteration_name $model_path $model_output_path $jsonl_input $json_output $reward_data_path $my_world_size $reward_model_path $json_output_wo $i
done
