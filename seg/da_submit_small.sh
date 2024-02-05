#!/bin/bash

#SBATCH  --output=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/LOGS/%j.out
#SBATCH  --error=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/LOGS/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH --account=staff
#SBATCH --constraint='titan_xp'
##SBATCH --constraint='titan_xp|geforce_rtx_2080_ti|geforce_gtx_1080_ti'
##SBATCH --constraint='geforce_rtx_2080_ti|titan_xp'

source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
conda activate mic
 
WORKPATH=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/

python run_experiments.py "$@" 