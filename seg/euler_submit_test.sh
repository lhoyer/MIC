#!/bin/bash

#SBATCH  --output=./LOGS_TEST/%j.out
#SBATCH  --error=./LOGS_TEST/%j.out
#SBATCH  --gpus=1
#SBATCH  --mem-per-cpu=5G
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=2
#SBATCH  --time=1-0

source /cluster/home/klanna/conda/conda/etc/profile.d/conda.sh
conda activate /cluster/project/cvl/klanna/conda-envs/mic-mmcv-full-prebuilt

WORKPATH=/cluster/work/cvl/klanna/MIC-results/ 

sh ./test_med.sh "$@" e