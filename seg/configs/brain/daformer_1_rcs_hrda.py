# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
datatag = '_ContrastFlip'
_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    f'../_base_/datasets/uda_brain_hcp1_full-hcp2_256x256{datatag}.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
# loss_name = 'DiceLoss'
loss_name = 'CrossEntropyLoss'
norm_cfg=False

seed = 0
# Modifications to Basic UDA

model = dict(
    norm_cfg=norm_cfg,
    type='HRDAEncoderDecoder',
    decode_head=dict(
        num_classes=15,
        loss_decode=dict(
            type=loss_name, use_sigmoid=False, loss_weight=1.0),
        type='HRDAHead',
        # Use the DAFormer decoder for each scale.
        single_scale_head='DAFormerHead',
        # Learn a scale attention for each class channel of the prediction.
        attention_classwise=True,
        # Set the detail loss weight $\lambda_d=0.1$.
        hr_loss_weight=0.1),
    # Use the full resolution for the detail crop and half the resolution for
    # the context crop.
    scales=[1, 0.5],
    # Use a relative crop size of 0.5 (=512/1024) for the detail crop.
    hr_crop_size=[128, 128],
    # Use LR features for the Feature Distance as in the original DAFormer.
    feature_scale=0.5,
    # Make the crop coordinates divisible by 8 (output stride = 4,
    # downscale factor = 2) to ensure alignment during fusion.
    crop_coord_divisible=8,
    # Use overlapping slide inference for detail crops for pseudo-labels.
    hr_slide_inference=True,
    # Use overlapping slide inference for fused crops during test time.
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[128, 128],
        crop_size=[256, 256]))

uda = dict(
    # Apply masking to color-augmented target images
    mask_mode='separatetrgaug',
    # Use the same teacher alpha for MIC as for DAFormer
    # self-training (0.999)
    mask_alpha='same',
    # Use the same pseudo label confidence threshold for
    # MIC as for DAFormer self-training (0.968)
    mask_pseudo_threshold='same',
    # Equal weighting of MIC loss
    mask_lambda=1,
    # Use random patch masking with a patch size of 64x64
    # and a mask ratio of 0.7
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True),
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    # imnet_feature_dist_lambda=0.005,
    # imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    # imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0)

class_temp=0.1
per_image=True
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=8, 
            class_temp=class_temp, 
            min_crop_ratio=0.5,
            per_image=per_image)
    ))

# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))

n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=1)
evaluation = dict(interval=100, metric='mDice')
# Meta Information for Result Analysis
norm_flag = '-norm' if norm_cfg else ''
name = f'brain_hcp1_full-hcp2{datatag}_daformer1{norm_flag}_{int(per_image)}rcs{class_temp:.1f}_{loss_name}'
exp = 'basic'
name_dataset = f'brain_hcp1_full-hcp2{datatag}'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'