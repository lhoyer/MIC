# ---------------------------------------------------------------
# Copyright (c) 2024 ETH Zurich, Anna Susmelj. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------


import torch
from torch.nn import Module
from ..builder import build_loss


class ContrastiveModule(Module):
    def __init__(self, loss_contrstive):
        super(ContrastiveModule, self).__init__()

        self.loss = build_loss(loss_contrstive)

        self.debug = False
        self.debug_output = {}

    def __call__(
        self, model, img, gt_semantic_seg, target_img, pseudo_label, pseudo_weight
    ):
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device

        contrast_loss = dict()

        mixed_img = torch.cat((img, target_img), dim=0)
        mixed_lbl = torch.cat((gt_semantic_seg, pseudo_label), dim=0).squeeze(1)
        gt_weight = torch.ones_like(gt_semantic_seg).squeeze(1).to(dev)
        mixed_weight = torch.cat((gt_weight, pseudo_weight), dim=0)

        mixed_logits, mixed_features = model.decode_head.forward(
            model.extract_feat(mixed_img), return_features=True
        )

        contrast_loss["loss_contrast"] = self.loss(
            mixed_logits, mixed_lbl, weight=mixed_weight, features=mixed_features
        )

        if self.debug:
            self.debug_output["Contrastive"] = model.debug_output

        return contrast_loss
