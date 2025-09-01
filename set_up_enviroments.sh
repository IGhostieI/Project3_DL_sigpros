#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 
#SBATCH --time=02:15:00
#SBATCH --job-name=set_up_enviroments
#SBATCH --output=set_up_enviroments.out

# Set up environment
uenv verbose cuda-11.8.0
uenv verbose cudnn-11.x-8.7.0 
uenv verbose miniconda3-py310

# Define the environment name as a variable
ENV_NAME="test_tf"

if ! conda info --envs | grep -q "$ENV_NAME"; then
    echo "Creating environment $ENV_NAME"
    conda create -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"

# Install base packages with conda
conda install -y numpy matplotlib scikit-learn scipy tqdm pydot graphviz packaging pandas pillow -c conda-forge -c anaconda

pip install tensorflow[and-cuda]==2.14.0 tensorflow-addons

# Verify TensorFlow can see the GPU
python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

conda deactivate

echo "Environment setup complete"