# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
datatag = ""
datatag = "_euler"
dataset = 'brain_abidec-hcp2'
num_classes=15

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/daformer_sepaspp_mitb5.py',
    # GTA->Cityscapes Data Loading
    f'../_base_/datasets/uda_{dataset}_256x256{datatag}.py',
    # Basic UDA Self-Training
    '../_base_/uda/dacs.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
# loss_name_teacher = 'DiceLoss'
# ========================================
# setting up the loss function for source
# ----------------------------------------
loss_name = 'CE'
loss_decode=dict(type='CrossEntropyLoss', 
                        use_sigmoid=False, 
                        loss_weight=1.0)
# ----------------------------------------
# loss_name = 'CEmem'
# loss_decode=dict(type='ContrastMemoryBankCELoss', 
#                         use_sigmoid=False, 
#                         memory_bank_size=2048,
#                         # memory_bank_size=512,
#                         ignore_label=-1,
#                         loss_weight=1.0)
# ----------------------------------------
# loss_name = 'CEmemIgnore'
# loss_decode=dict(type='ContrastMemoryBankCELoss', 
#                         use_sigmoid=False, 
#                         memory_bank_size=2048,
#                         ignore_label=0,
#                         loss_weight=1.0)
# ========================================
norm_net=False
model = dict(decode_head=dict(num_classes=num_classes, 
                              loss_decode=loss_decode), 
                                norm_cfg=norm_net)

seed = 0

# MIC Parameters
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
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0)

class_temp=0.1
per_image=False
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        # Rare Class Sampling
        rare_class_sampling=dict(
            min_pixels=4, 
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
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mDice')
# Meta Information for Result Analysis
norm_flag = '-norm' if norm_net else ''
name = f'{dataset}{datatag}_daformersepaspp5{norm_flag}_mic_{loss_name}'
exp = 'basic'
name_dataset = f'{dataset}{datatag}'
name_architecture = 'daformer_sepaspp_mitb5'
name_encoder = 'mitb5'
name_decoder = 'daformer_sepaspp'
name_uda = 'dacs_a999_fd_things_rcs0.01_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'