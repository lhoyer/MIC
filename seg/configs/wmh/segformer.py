# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# WMH datasets
datatag = ""
datatag = "_noph_v2_euler"
# datatag = "_euler"
# dataset = "wmh_nuhs-umc"
dataset = "wmh_umc-nuhs"
num_classes = 2

_base_ = [
    "../_base_/default_runtime.py",
    # DAFormer Network Architecture
    "../_base_/models/segformer_r101.py",
    # GTA->Cityscapes Data Loading
    f"../_base_/datasets/uda_{dataset}_256x256{datatag}.py",
    # Basic UDA Self-Training
    "../_base_/uda/dacs.py",
    # AdamW Optimizer
    "../_base_/schedules/adamw.py",
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    "../_base_/schedules/poly10warm_med.py",
]

model = dict(decode_head=dict(num_classes=num_classes))

seed = 0
# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    # imnet_feature_dist_lambda=0.005,
    # imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    # imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
)

class_temp = 0.1
per_image = False
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=4, class_temp=class_temp, min_crop_ratio=0.5, per_image=per_image
        )
    ),
)
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    # lr=6e-05,
    lr=6e-04,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
        )
    ),
)
n_gpus = 1
runner = dict(type="IterBasedRunner", max_iters=10000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)
evaluation = dict(interval=500, metric="mDice")

# Meta Information for Result Analysis
exp = "basic"
name_dataset = f"{dataset}{datatag}"
name_architecture = "segformer_r101"
name_encoder = "ResNetV1c"
name_decoder = "SegFormerHead"
name_uda = "dacs"
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x2_10k"
name = f"{dataset}{datatag}_{name_architecture}"
