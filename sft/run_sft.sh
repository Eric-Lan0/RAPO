# bash ./sft/run_sft.sh

# You can adjust the training parameters in ./sft/sft.py
# accelerate launch ./sft/sft.py

# Train with deepspeed stage3 
# You may need to adjust ./configs/zero3.yaml, especially the num_processes (the number of GPUs) according to your environment
accelerate launch --config_file ./configs/zero3.yaml ./sft/sft.py \
    --dataset_name="/apdcephfs_us/share_300814644/user/ericglan/data/SFT-OpenHermes-2.5-Standard" \
    