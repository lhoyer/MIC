# Obtained from: https://github.com/lhoyer/HRDA
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

data_root = '/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/brain/'

# dataset settings
dataset_type = 'BrainDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (256, 256)
# img_scale = (320, 320)
img_scale = (256, 256)
source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale),
    # dict(type='ContrastFlip', data_aug_ratio=1.0),  
    # dict(type='CLAHE'),
    dict(type='ElasticTransformation', data_aug_ratio=0.25),  
    dict(type='StructuralAug', data_aug_ratio=0.25),
    # dict(type='MedPhotoMetricDistortion', data_aug_ratio=0.25),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale),
    # dict(type='StructuralAug', data_aug_ratio=0.25),
    # dict(type='RandomCrop', crop_size=crop_size),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='MedPhotoMetricDistortion', data_aug_ratio=0.25),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='BrainDataset',
            data_root=f'{data_root}/hcp1/',
            img_dir='images/train',
            ann_dir='labels/train',
            pipeline=source_train_pipeline),
        target=dict(
            type='BrainDataset',
            data_root=f'{data_root}/hcp2/',
            img_dir='images/train',
            ann_dir='labels/train',
            pipeline=target_train_pipeline)),
    val=dict(
        type='BrainDataset',
        data_root=f'{data_root}/hcp2/',
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=test_pipeline),
    test=dict(
        type='BrainDataset',
        data_root=f'{data_root}/hcp2/',
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=test_pipeline))