#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 
#SBATCH --time=10:15:00
#SBATCH --job-name=tensorflow_test_job
#SBATCH --output=gpu_test.out
 
# Activate environment
uenv verbose cuda-12.2.2; uenv verbose cudnn-12.x-8.9.7
uenv  verbose miniconda3-py310
conda activate slurm_pytorch_env_py310
# Run the Python script that uses the GPU
cd /home/stud/casperc/bhome/Project3_DL_sigpros
echo "Running the job"
python -u tf_test.py
echo "Done with the job"