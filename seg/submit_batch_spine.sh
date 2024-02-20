#!/bin/bash
sbatch ./euler_submit.sh --config configs/spine/segformer_fda.py
sbatch ./euler_submit.sh --config configs/spine/segformer_fda.py

sbatch ./euler_submit.sh --config configs/spine/segformer_gan.py
sbatch ./euler_submit.sh --config configs/spine/segformer_gan.py

sbatch ./euler_submit.sh --config configs/spine/segformer.py
sbatch ./euler_submit.sh --config configs/spine/segformer.py

sbatch ./euler_submit.sh --config configs/spine/segformer_colormix.py
sbatch ./euler_submit.sh --config configs/spine/segformer_colormix.py

sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_2.py
sbatch ./euler_submit.sh --config configs/spine/segformer_colormix_2.py

sbatch ./euler_submit.sh --config configs/spine/segformer_src.py