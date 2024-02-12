#!/bin/bash

#SBATCH  --output=./LOGS/%j.out
#SBATCH  --error=./LOGS/%j.out
#SBATCH  --gpus=1
#SBATCH  --mem-per-cpu=5G
#SBATCH  --ntasks=1
#SBATCH  --cpus-per-task=2
#SBATCH  --time=1-0

source /cluster/home/klanna/conda/conda/etc/profile.d/conda.sh
conda activate /cluster/project/cvl/klanna/conda-envs/mic-mmcv-full-prebuilt

WORKPATH=/cluster/work/cvl/klanna/MIC-results/ 

# python -u run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py
python -u run_experiments.py "$@" --out_path "/cluster/work/cvl/klanna/MIC-results"


# sbatch --time=1-0 --mem-per-cpu=40G --gpus=1  --ntasks=1 --cpus-per-task=4  ./euler_submit.sh

# srun --pty bash --time=4-0 --mem-per-cpu=40G --gpus=1  --ntasks=1 --cpus-per-task=4

# sbatch --time=1-0 --mem-per-cpu=40G --gpus=1  --ntasks=1 --cpus-per-task=4  ./euler_submit.sh --config configs/hcp/danet_colormix.py