#!/bin/bash

#SBATCH  --output=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/LOGS/%j.out
#SBATCH  --error=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=80G
##SBATCH --account=bmic
##SBATCH --time=300

WORKPATH=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate mic

python run_experiments.py "$@" 
# --work-dir $WORKPATH

# --config configs/mic/gtaHR2csHR_mic_hrda.py
# --config configs/brain/test.py
