# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer='BN'):
        super().__init__()

        if norm_layer == 'IN':
            norm_layer = nn.InstanceNorm2d
        elif norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_layer=norm_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_layer=norm_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
     
class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHeadHR(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, **kwargs):
        super(SegFormerHeadHR, self).__init__(
            input_transform='multiple_select', **kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.linear_c = {}
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.inc = DoubleConv(3, self.in_channels[0], norm_layer='BN')
        self.down = Down(self.in_channels[0], self.in_channels[0], "BN")
        self.up1 = Up(self.in_channels[0] + embedding_dim, embedding_dim, 'BN', bilinear=True, scale_factor=2)
        self.up2 = Up(self.in_channels[0] + embedding_dim, embedding_dim, 'BN', bilinear=True, scale_factor=2)

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs[:-1]
        img = inputs[-1]
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f.shape)

        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c

        x1 = self.inc(img)
        x2 = self.down(x1)
        x = self.up2(x, x2)
        x = self.up1(x, x1)

        x = self.linear_pred(x)

        return x
