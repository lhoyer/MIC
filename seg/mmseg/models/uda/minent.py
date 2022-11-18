# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The entropy minimization is based on: https://github.com/valeoai/ADVENT
# Copyright (c) 2019 Valeo. Licensed under the Apache License, Version 2.0
# A copy of the license is available at resources/license_advent

import os

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.utils.dacs_transforms import denorm, get_mean_std
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.ops import resize


def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (
        n * h * w * np.log2(c))


def entropy_map(v):
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30)), dim=1) / np.log2(c)


@UDA.register_module()
class MinEnt(UDADecorator):

    def __init__(self, **cfg):
        super(MinEnt, self).__init__(**cfg)
        self.lambda_ent = cfg['lambda_ent']
        self.mask_mode = cfg['mask_mode']

        if self.mask_mode is not None:
            self.mic = MaskingConsistencyModule(require_teacher=True, cfg=cfg)

        self.local_iter = 0

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        self.update_debug_state()
        seg_debug = {}

        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        # train on source
        src_losses = dict()
        pred = self.model.forward_with_aux(img, img_metas)
        loss = self.model.decode_head.losses(pred['main'], gt_semantic_seg)
        if self.get_model().debug:
            self.get_model().process_debug(img, img_metas)
            seg_debug['Source'] = self.get_model().debug_output
            self.get_model().debug_output = {}
        if isinstance(self.model, HRDAEncoderDecoder):
            self.model.decode_head.reset_crop()
        src_losses.update(add_prefix(loss, 'decode'))
        if self.model.with_auxiliary_head:
            loss_aux = self.model.auxiliary_head.losses(
                pred['aux'], gt_semantic_seg)
            src_losses.update(add_prefix(loss_aux, 'aux'))
        src_loss, src_log_vars = self._parse_losses(src_losses)
        src_loss.backward()

        # entropy minimization on target
        trg_losses = dict()
        pred_trg = self.model.forward_with_aux(target_img, target_img_metas)
        if isinstance(self.model, HRDAEncoderDecoder):
            self.model.decode_head.reset_crop()
            for k in pred.keys():
                pred_trg[k] = pred_trg[k][0]
                assert self.model.feature_scale == 0.5
                pred_trg[k] = resize(
                    input=pred_trg[k],
                    size=[
                        int(e * self.model.feature_scale)
                        for e in img.shape[2:]
                    ],
                    mode='bilinear',
                    align_corners=self.model.align_corners)
        for k in pred_trg.keys():
            # remember to have word 'loss' in key
            trg_losses[f'ent.loss.{k}'] = self.lambda_ent[k] * entropy_loss(
                F.softmax(pred_trg[k], dim=1))
        trg_loss, trg_log_vars = self._parse_losses(trg_losses)
        trg_loss.backward()

        # masking consistency
        masked_log_vars = dict()
        if self.mic is not None:
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            masked_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            batch_size = img.shape[0]
            means, stds = get_mean_std(img_metas, target_img.device)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_ent = entropy_map(F.softmax(pred_trg['main'], dim=1))
            for j in range(batch_size):
                rows, cols = 2, 3
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[1][1],
                    torch.argmax(pred_trg['main'][j], dim=0),
                    'Target Seg',
                    cmap='cityscapes')
                vmin = torch.min(vis_ent[j]).item()
                vmax = torch.max(vis_ent[j]).item()
                subplotimg(
                    axs[1][2],
                    vis_ent[j],
                    f'Target Ent {vmin:.2E}, {vmax:.2E}',
                    cmap='viridis')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

            if seg_debug['Source'] is not None and seg_debug:
                for j in range(batch_size):
                    rows = len(seg_debug)
                    cols = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k1][k2],
                                **prepare_debug_out(f'{n1} {n2}', out[j],
                                                    means, stds))
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()

        self.local_iter += 1

        return {**src_log_vars, **trg_log_vars}
