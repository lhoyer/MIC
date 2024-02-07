#!/bin/bash

# srun --time 480 --account=staff --partition=gpu.debug --mem=50G --gres=gpu:1 --pty bash -i 
# srun --time 480 --partition=gpu.debug --gres=gpu:1 --mem=80G --pty bash -i

# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate mic

# source /scratch_net/tinto/klanna/conda/etc/profile.d/conda.sh
# conda activate pytcu11-gpu

WANDB_MODE=disabled
WORKPATH=/usr/bmicnas02/data-biwi-01/klanna_data/results/MIC/

# python run_experiments.py --config configs/brain/debug.py

# python run_experiments.py --config configs/brain/baseline_cnn.py

# python run_experiments.py --config configs/brain/segformer_rcs.py

# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py

# python run_experiments.py --config configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0_contrastive.py 

# python run_experiments.py --config configs/hrda/gtaHR2csHR_hrda.py

# python run_experiments.py --config configs/whitematter/daformer_mic.py

# python run_experiments.py --config configs/whitematter/segformer.py


# python run_experiments.py --config configs/brain/daformer_mic_colormix_source.py

python run_experiments.py --config configs/brain/segformer_colormix_source.py

# python run_experiments.py --config configs/whitematter/segformer_colormix_source.py
