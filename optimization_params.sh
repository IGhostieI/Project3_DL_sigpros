#!/bin/bash
#SBATCH --partition=cpu64
#SBATCH --time=15:00:00
#SBATCH --job-name=grid_search
#SBATCH --output=/home/stud/casperc/bhome/Project3_DL_sigpros/Optimization_param_analysis/optm2.out

uenv verbose miniconda3-py310
conda activate tensorflow_env_py310

python -u ftol_maxiter_order_choice.py
echo "Done with the job"