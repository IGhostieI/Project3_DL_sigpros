#!/bin/bash
cd /home/Project1\(DeepFit\)/
# Set up environment
uenv  verbose miniconda3-py311
uenv verbose cuda-11.8.0

if ! conda info --envs | grep -q 'pytorch_env_py311'; then
    echo "Creating environment pytorch_env_py311"
    conda create -n pytorch_env_py311 python=3.11.5 -y
fi
conda activate pytorch_env_py311
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda deactivate

if ! conda info --envs | grep -q 'tensorflow_env_py311'; then
    echo "Creating environment tensorflow_env_py311"
    conda create -n tensorflow_env_py311 python=3.11.5 -y
fi
conda activate tensorflow_env_py311
pip3 install -r requirements.txt
python3 -m pip install tensorflow[and-cuda]
conda deactivate

uenv remove cuda-11.8.0

uenv verbose cuda-12.3.2 cudnn-12.x-9.0.0
# Check if the environment exists, if not, create it
if ! conda info --envs | grep -q 'slurm_pytorch_env_py311'; then
    conda create -n slurm_pytorch_env_py311 python=3.11.5 -y
fi

conda activate slurm_pytorch_env_py311
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda deactivate

# Check if the environment exists, if not, create it
if ! conda info --envs | grep -q 'slurm_tensorflow_env_py311'; then
    echo "Creating environment slurm_tensorflow_env_py311"
    conda create -n slurm_tensorflow_env_py311 python=3.11.5 -y
fi
conda activate slurm_tensorflow_env_py311
pip3 install -r requirements.txt
pip3 install tensorflow[and-cuda]
conda deactivate

uenv remove cuda-12.3.2 cudnn-12.x-9.0.0
uenv remove miniconda3-py311

echo "Environment setup complete"