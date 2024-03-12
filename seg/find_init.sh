#!/bin/bash

#SBATCH  --output=./LOGS_init/%j.out
#SBATCH  --error=./LOGS_init/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=80G
##SBATCH --account=bmic
##SBATCH --time=300

WORKPATH=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate mic

python tools/find_colormix_init.py "$@" 