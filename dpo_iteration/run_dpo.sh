conda activate rlhflow
# bash ./dpo_iteration/run_dpo.sh
model_path="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/sft_checkpoint/LLaMA3-SFT"
initial_model="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/sft_checkpoint/LLaMA3-SFT"
# model_path=meta-llama/Meta-Llama-3-8B-Instruct
# initial_model=meta-llama/Meta-Llama-3-8B-Instruct
# mkdir checkpoint
accelerate launch --config_file ./configs/zero3.yaml ./dpo_iteration/run_dpo.py \
    --run_name="rlhflow_iter1" \
    --output_dir="./checkpoint/test/rlhflow_iter1" \
    --model_name_or_path=$model_path \
    --ref_model=$initial_model \
    --learning_rate=2e-7 \
    --max_steps=1200 \
    --choose_type="max_min" \
    --train_dir="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/checkpoint/test/data_1_LLaMA3_iter1_reward.json" \
    --eval_dir="/apdcephfs_us/share_300814644/user/ericglan/Online-RLHF/checkpoint/test/data_1_LLaMA3_iter1_reward.json" \
    --loss_type="sigmoid" \
    --lr_scheduler_type="cosine"
