#!/bin/bash

source ~/.bashrc

# Activate the virtual environment
# source /path/to/venv/bin/activate
#source /home/brionyf/anaconda3/envs/anomalib_env/lib/python3.8/venv/scripts/common/activate

eval "$('/home/brionyf/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#eval "$(conda shell.bash hook)"
conda deactivate
conda activate anomalib_env

# Change to the directory containing the Python script
cd '/home/brionyf/Desktop/Code/03 Pylon/AKL Pylon'

# Run the Python script
python pylon_viewer.py

$SHELL

# Source: https://linuxconfig.org/how-to-use-a-bash-script-to-run-your-python-scripts


