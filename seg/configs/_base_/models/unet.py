# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications:
# - BN instead of SyncBN
# - Replace MiT with ResNet backbone
# This work is licensed under the NVIDIA Source Code License
# A copy of the license is available at resources/license_segformer


# model settings
norm_cfg = dict(type='BN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='UNetEnc',
        layers=(64, 128, 256, 512, 1024),
        bilinear=True,
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='UNetDec',
        layers=(64, 128, 256, 512, 1024),
        bilinear=True,
        norm_cfg=norm_cfg,
        num_classes=15,
        align_corners=False,
        decoder_params=dict(embed_dim=768, conv_kernel_size=1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
