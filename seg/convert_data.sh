#!/bin/bash

# source /itet-stor/klanna/net_scratch/conda/etc/profile.d/conda.sh
# conda activate pytcu11

PATH=/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/datasets/self-driving/

python tools/convert_datasets/gta.py /itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/datasets/self-driving//gta --nproc 8

python tools/convert_datasets/cityscapes.py /itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/datasets/self-driving/cityscapes/gtFine_trainvaltest --nproc 8