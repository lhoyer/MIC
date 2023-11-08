#!/bin/bash

# srun --time 120 --account=staff --partition=gpu.debug --gres=gpu:1 --pty bash -i
# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate pytcu11

python run_experiments.py --config configs/brain/debug.py