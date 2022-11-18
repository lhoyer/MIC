# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import torch

from mmseg.ops import resize


def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(
            W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask
