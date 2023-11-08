#!/bin/bash

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
##SBATCH --account=bmic

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate pytcu11
 
WORKPATH=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/

python run_experiments.py "$@" 
# --work-dir $WORKPATH

# --config configs/mic/gtaHR2csHR_mic_hrda.py
# --config configs/brain/test.py
