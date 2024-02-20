#!/bin/bash

sbatch ./euler_submit.sh --config configs/hcp/segformer_fda.py
sbatch ./euler_submit.sh --config configs/hcp/segformer_fda.py

# sbatch ./euler_submit.sh --config configs/hcp/segformer_gan.py
# sbatch ./euler_submit.sh --config configs/hcp/segformer_gan.py

sbatch ./euler_submit.sh --config configs/hcp/segformer.py
sbatch ./euler_submit.sh --config configs/hcp/segformer.py

# sbatch ./euler_submit.sh --config configs/hcp/segformer_colormix.py
# sbatch ./euler_submit.sh --config configs/hcp/segformer_colormix.py

sbatch ./euler_submit.sh --config configs/hcp/segformer_colormix.py
sbatch ./euler_submit.sh --config configs/hcp/segformer_colormix.py

sbatch ./euler_submit.sh --config configs/hcp/segformer_colormix_2.py
sbatch ./euler_submit.sh --config configs/hcp/segformer_colormix_2.py

