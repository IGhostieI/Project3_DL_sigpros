#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=02:15:00
#SBATCH --job-name=auto_encoder_setup
#SBATCH --output=auto_encoder_setup.out
 
# Set up environment
uenv verbose cuda-12.3.2 cudnn-12.x-9.0.0
uenv miniconda3-py311
conda create -n slurm_pytorch_env_py311 -y
conda activate slurm_pytorch_env_py311
pip install -r requirements_slurm.txt 
pip install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 -y
pip freeze 
