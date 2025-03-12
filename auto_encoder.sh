#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100 
#SBATCH --time=10:15:00
#SBATCH --job-name=pytorch_mnist
#SBATCH --output=auto_encoder_test_01.out
 
# Activate environment
uenv verbose cuda-12.3.2 cudnn-12.x-9.0.0
uenv miniconda3-py311
conda activate slurm_pytorch_env_py311
# Run the Python script that uses the GPU
python -u autoencoder_small_data.py
echo "Done with the job"