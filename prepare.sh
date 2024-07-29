export https_proxy=http://9.21.0.122:11113
export http_proxy=http://9.21.0.122:11113
export ftp_proxy=http://9.21.0.122:11113

#####################################################################
#  Inference Environment
#####################################################################
# conda create -n vllm python=3.10.9
# conda activate vllm
conda create -p /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/vllm python=3.10.9
conda activate /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/vllm

pip install datasets
# The following code is tested for CUDA12.0-12.2. You may need to update the torch and flash-attention sources according to your own CUDA version
pip3 install torch==2.1.2 torchvision torchaudio
pip install https://github.com/vllm-project/vllm/releases/download/v0.4.0/vllm-0.4.0-cp310-cp310-manylinux1_x86_64.whl 
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install numpy==1.26.4
pip install transformers==4.43.3
pip install accelerate==0.33.0
pip install deepspeed


#####################################################################
#  Training Environment
#####################################################################
# conda create -n rlhflow python=3.10.9
# conda activate rlhflow
conda create -p /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/rlhflow python=3.10.9
conda activate /apdcephfs_us/share_300814644/user/ericglan/.conda/envs/rlhflow

# git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout d17fd7cd3b71c6a7bf7af34d8dc73135bb7ea8e9
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.7/flash_attn-2.5.7+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install numpy==1.26.4
pip install transformers==4.43.3
pip install accelerate==0.33.0
pip install trl==0.9.6

pip install wandb
cd ..

# huggingface token: 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
wandb login
huggingface-cli login


#####################################################################
#  Evaluation
#####################################################################
# git clone https://github.com/EleutherAI/lm-evaluation-harness.git