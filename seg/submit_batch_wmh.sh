#!/bin/bash

# sbatch ./euler_submit.sh --config configs/wmh/segformer_fda.py
# sbatch ./euler_submit.sh --config configs/wmh/segformer_fda.py

# sbatch ./euler_submit.sh --config configs/wmh/segformer_gan.py
# sbatch ./euler_submit.sh --config configs/wmh/segformer_gan.py

# sbatch ./euler_submit.sh --config configs/wmh/segformer.py
# sbatch ./euler_submit.sh --config configs/wmh/segformer.py

sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix.py
sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix.py

sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix_col.py
sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix_col.py

# sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix_bcg_col.py
# sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix_bcg_col.py

# sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix_bcg.py
# sbatch ./euler_submit.sh --config configs/wmh/segformer_colormix_bcg.py
