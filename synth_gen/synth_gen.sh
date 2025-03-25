#!/bin/bash
#SBATCH --partition=cpu64 
#SBATCH --time=23:00:00
#SBATCH --job-name=synth_gen
#SBATCH --output=/home/stud/casperc/bhome/Project3_DL_sigpros/synth_gen/synth_gen.out

uenv verbose miniconda3-py310
conda activate slurm_tensorflow_env_py310

python -u synthetic_augmented_data_generation_standard_amplitude.py
echo "Done with the job"