# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# datatag = '_ContrastFlip_v8'
# # dataset='brain_hcp1-abidec'
# dataset='hcp1-abidecal'
# num_classes=15

datatag = '_ContrastFlip_v8'
dataset = 'wmh_umc-nuhs'
# dataset = 'wmh_vu-nuhs'
# dataset = 'wmh_nuhs-umc'
num_classes=2

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/segformer_r101.py',
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
# loss_name = 'CE'
# loss_decode=dict(type='CrossEntropyLoss', 
#                         use_sigmoid=False, 
#                         loss_weight=1.0)
# ----------------------------------------
# loss_name = 'CEmemIgnore'
# loss_decode=dict(type='ContrastMemoryBankCELoss', 
#                         use_sigmoid=False, 
#                         memory_bank_size=2048,
#                         ignore_label=0,
#                         loss_weight=1.0)
# ----------------------------------------
loss_name = 'CEmem'
loss_decode=dict(type='ContrastMemoryBankCELoss', 
                        use_sigmoid=False, 
                        memory_bank_size=2048,
                        ignore_label=255,
                        loss_weight=1.0)
# ----------------------------------------
# loss_name = 'Contrast'
# loss_decode=dict(type='ContrastCELoss', 
#                         use_sigmoid=False, 
#                         ignore_label=-1,
#                         loss_weight=1.0)
# ========================================

# ========================================
# setting up the loss function for target
loss_name_teacher = 'CE'
loss_decode_teacher=dict(type='CrossEntropyLoss', 
                        use_sigmoid=False, 
                        loss_weight=1.0)
# ========================================

norm_cfg=True
seed = 0
model = dict(decode_head=dict(num_classes=num_classes, 
                              loss_decode=loss_decode,
                                loss_decode_teacher=loss_decode_teacher), 
                                norm_cfg=norm_cfg)
# Modifications to Basic UDA

uda = dict(
    # Apply masking to color-augmented target images
    # mask_mode='separatetrgaug',
    # # Use the same teacher alpha for MIC as for DAFormer
    # # self-training (0.999)
    # mask_alpha='same',
    # # Use the same pseudo label confidence threshold for
    # # MIC as for DAFormer self-training (0.968)
    # mask_pseudo_threshold='same',
    # # Equal weighting of MIC loss
    # mask_lambda=1,
    # # Use random patch masking with a patch size of 64x64
    # # and a mask ratio of 0.7
    # mask_generator=dict(
    #     type='block', mask_ratio=0.7, mask_block_size=8, _delete_=True),
    # # Increased Alpha
    # mix='class',
    mix=None,
    burnin=100,
    alpha=0.999,
    enable_contrastive=False,
    # contrastive_loss=dict(type='ContrastMemoryBankMixCELoss', 
    #     use_sigmoid=False,         
    #     loss_weight=1.0,
    #     temperature=0.1,
    #     base_temperature=0.1,
    #     ignore_label=-1,
    #     memory_bank_size=2048,
    #     max_views=8),
    # Thing-Class Feature Distance
    # imnet_feature_dist_lambda=0.005,
    # imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    # imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0)

class_temp=0.1
per_image=False
data = dict(
    samples_per_gpu=8,
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
mix_flag = '-mix' if uda['mix'] is not None else ''
burnin_flag = f'-burnin{uda["burnin"]}' if uda['burnin'] > 0 else ''
contrast_flag = '-contrast' if uda['enable_contrastive'] else ''
name = f'{dataset}{datatag}_segformer{norm_flag}_{int(per_image)}rcs{class_temp:.1f}_{loss_name}-{loss_name_teacher}{mix_flag}{burnin_flag}{contrast_flag}'
exp = 'basic'
name_dataset = f'{dataset}{datatag}'
name_architecture = 'segformer_r101'
name_encoder = 'ResNetV1c'
name_decoder = 'SegFormerHead'
name_uda = 'dacs'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'