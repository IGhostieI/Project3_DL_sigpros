#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 
#SBATCH --time=02:15:00
#SBATCH --job-name=set_up_enviroments
#SBATCH --output=tf_setup.out

# Set up environment
uenv verbose cuda-11.8.0
uenv verbose cudnn-11.x-8.7.0 
uenv verbose miniconda3-py310

conda env create -f /home/stud/casperc/bhome/Project3_DL_sigpros/enviroment.yml

conda activate tf_env_py310

pip check
pipdeptree -w fail
conda-tree deps numpy
