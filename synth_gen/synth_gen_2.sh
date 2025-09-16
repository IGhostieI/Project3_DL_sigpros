#!/bin/bash
#SBATCH --partition=cpu64 
#SBATCH --time=23:00:00
#SBATCH --job-name=synth_gen_2
#SBATCH --output=/home/stud/casperc/bhome/Project3_DL_sigpros/synth_gen/synth_gen_2.out

uenv verbose miniconda3-py310
conda activate tf_DL_py310 

python -u /home/stud/casperc/bhome/Project3_DL_sigpros/synthetic_augmented_data_generation_standard_amplitude.py
echo "Done with the job"