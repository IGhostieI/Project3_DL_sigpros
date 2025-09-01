#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 
#SBATCH --time=10:15:00
#SBATCH --job-name=DL_fit1
#SBATCH --output=DL_fit1.out

# Activate environment
uenv verbose cuda-11.8.0
uenv verbose cudnn-11.x-8.7.0 
uenv verbose miniconda3-py310
conda activate slurm_tensorflow_env_py310
# Run the Python script that uses the GPU
echo "Running the job"
python -u cnn_autoencoder_tf.py
echo "Done with the job"