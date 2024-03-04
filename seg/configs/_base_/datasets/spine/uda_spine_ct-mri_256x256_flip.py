# Obtained from: https://github.com/lhoyer/HRDA
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

data_root='/itet-stor/klanna/bmicdatasets_bmicnas02/Sharing/klanna/da_data/lumbarspine/'

# dataset settings
dataset_type = 'SpineCTDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
# img_scale = (320, 320)
img_scale = (256, 256)
source_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale),
    dict(type='ElasticTransformation', data_aug_ratio=0.25),  
    dict(type='ContrastFlip', data_aug_ratio=1.0, shift=0.40618, scale=-0.30236),
    dict(type='StructuralAug', data_aug_ratio=0.25),
    dict(type='RandomFlip', prob=0.0),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale),
    dict(type='ElasticTransformation', data_aug_ratio=0.25),
    dict(type='StructuralAug', data_aug_ratio=0.25),
    dict(type='RandomFlip', prob=0.0),
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='SpineCTDataset',
            data_root=f'{data_root}/VerSe/',
            img_dir='images/train',
            ann_dir='labels/train',
            pipeline=source_train_pipeline),
        target=dict(
            type='SpineMRIDataset',
            data_root=f'{data_root}/MRSpineSegV/',
            img_dir='images/train',
            ann_dir='labels/train',
            pipeline=target_train_pipeline)),
    val=dict(
        type='SpineMRIDataset',
        data_root=f'{data_root}/MRSpineSegV/',
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=test_pipeline),
    test=dict(
        type='SpineMRIDataset',
        data_root=f'{data_root}/MRSpineSegV/',
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=test_pipeline))
