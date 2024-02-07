# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import MaskingConsistencyModule
from mmseg.models.uda.contrastive_module import ContrastiveModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (
    denorm,
    denorm_,
    renorm_,
    get_class_masks,
    get_mean_std,
    strong_transform,
    ClasswiseMultAugmenter,
)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio
from mmseg.models.utils.wandb_log_images import WandbLogImages

from torch import nn
import wandb
from torchvision.utils import make_grid
from PIL import Image

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type
        )

    return norm


@UDA.register_module()
class DACS(UDADecorator):
    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg["max_iters"]
        self.source_only = cfg["source_only"]
        self.alpha = cfg["alpha"]
        self.pseudo_threshold = cfg["pseudo_threshold"]
        self.psweight_ignore_top = cfg["pseudo_weight_ignore_top"]
        self.psweight_ignore_bottom = cfg["pseudo_weight_ignore_bottom"]
        self.fdist_lambda = cfg["imnet_feature_dist_lambda"]
        self.fdist_classes = cfg["imnet_feature_dist_classes"]
        self.fdist_scale_min_ratio = cfg["imnet_feature_dist_scale_min_ratio"]
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg["mix"]
        self.blur = cfg["blur"]
        self.color_jitter_s = cfg["color_jitter_strength"]
        self.color_jitter_p = cfg["color_jitter_probability"]
        self.mask_mode = cfg["mask_mode"]
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg["print_grad_magnitude"]
        self.enable_contrastive = (
            cfg["enable_contrastive"] if "enable_contrastive" in cfg else False
        )
        self.burnin = cfg["burnin"] if "burnin" in cfg else 0
        self.color_mix = cfg["color_mix"] if "color_mix" in cfg else dict(type='none')

        # assert self.mix == 'class'
        if self.mix != "class":
            print("Mixing switched off!")

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg["model"])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg["model"]))
        else:
            self.imnet_model = None

        if self.enable_contrastive:
            self.contrastive = ContrastiveModule(cfg["contrastive_loss"])

        if self.color_mix["type"] != "none":
            num_classes = self.get_model().decode_head.num_classes
            self.contrast_flip = ClasswiseMultAugmenter(
                num_classes, self.color_mix["norm_type"], 
                suppress_bg=self.color_mix["suppress_bg"],
            )       
            self.criterion = nn.MSELoss()

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(
            self.get_ema_model().parameters(), self.get_model().parameters()
        ):
            if not param.data.shape:  # scalar tensor
                ema_param.data = (
                    alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
                )
            else:
                ema_param.data[:] = (
                    alpha_teacher * ema_param[:].data[:]
                    + (1 - alpha_teacher) * param[:].data[:]
                )

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

        log_vars.pop("loss", None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch["img_metas"]))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if (
            isinstance(self.get_model(), HRDAEncoderDecoder)
            and self.get_model().feature_scale
            in self.get_model().feature_scale_all_strs
        ):
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img, False)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled, HRDAEncoderDecoder.last_train_crop_box[s]
                        )
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = (
                        downscale_label_ratio(
                            gt_rescaled,
                            scale_factor,
                            self.fdist_scale_min_ratio,
                            self.num_classes,
                            255,
                        )
                        .long()
                        .detach()
                    )
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s], fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]

            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]

                gt_rescaled = (
                    downscale_label_ratio(
                        gt,
                        scale_factor,
                        self.fdist_scale_min_ratio,
                        self.num_classes,
                        255,
                    )
                    .long()
                    .detach()
                )
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(
                    feat[lay], feat_imnet[lay], fdist_mask
                )
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses({"loss_imnet_feat_dist": feat_dist})
        feat_log.pop("loss", None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device
        )
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, : self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom :, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def intensity_normalization(self, img_original, gt_semantic_seg, means, stds):
        # estimate tgt intensities using GT segmenation masks
        
        img_segm_hist = self.contrast_flip.color_mix(
            img_original, gt_semantic_seg, means, stds
        )     

        # img, loss_val = self.contrast_flip.optimization_step(img_original, img_segm_hist, gt_semantic_seg)

        # update normalization net
        norm_net = self.get_model().normalization_net
        img_polished = norm_net(img_original[:, 0, :, :].unsqueeze(1))        

        if self.color_mix['suppress_bg']:
             ## automatically detect background value
            # background_val = img_original[0, 0, 0, 0].item()
            # foreground_mask = img_original[:, 0, :, :].unsqueeze(1) > 0
            # background_mask = img_original[:, 0, :, :].unsqueeze(1) == background_val
            
            foreground_mask = gt_semantic_seg > 0
            background_mask = gt_semantic_seg == 0

            norm_loss = self.criterion(img_polished[foreground_mask], img_segm_hist[foreground_mask])
        else:
            norm_loss = self.criterion(img_polished, img_segm_hist)

        norm_loss.backward(retain_graph=False)
        
        img = img_polished.detach()
        del img_polished

        if self.color_mix['suppress_bg']:
            img[background_mask] = img_segm_hist[background_mask]
            # img[background_mask] = img_original[:, 0, :, :].unsqueeze(1)[background_mask].mean().item()

        img = img.repeat(1, 3, 1, 1)


        if self.local_iter % 20 == 0:
            # for i in range(self.contrast_flip.n_classes):
            #     wandb.log({f"Class_{i+1} src": self.contrast_flip.source_mean[i, 0].item()}, step=self.local_iter+1)
            #     wandb.log({f"Class_{i+1} tgt": self.contrast_flip.target_mean[i, 0].item()}, step=self.local_iter+1)

            # for name, param in self.contrast_flip.normalization_net.named_parameters():
                # wandb.log({name: param.data.item()}, step=self.local_iter+1)
            # wandb.log({'loss': loss_val}, step=self.local_iter+1)
            wandb.log({'loss': norm_loss.item()}, step=self.local_iter+1)

            vis_img = torch.clamp(denorm(img_original, means, stds), 0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()
            vis_trg_img = torch.clamp(denorm(img_segm_hist, means, stds), 0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()
            vis_mixed_img = torch.clamp(denorm(img, means, stds), 0, 1).cpu().permute(0, 2, 3, 1)[0].numpy()

            wandb.log({"Augmentation": wandb.Image(
                np.concatenate([vis_img, vis_trg_img, vis_mixed_img], axis=1))})       

        return img
    
    def forward_train(
        self,
        img,
        img_metas,
        gt_semantic_seg,
        target_img,
        target_img_metas,
        rare_class=None,
        valid_pseudo_mask=None,
    ):
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

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            "mix": None,
            "color_jitter": random.uniform(0, 1),
            "color_jitter_s": self.color_jitter_s,
            "color_jitter_p": self.color_jitter_p,
            "blur": random.uniform(0, 1) if self.blur else 0,
            "mean": means[0].unsqueeze(0),  # assume same normalization
            "std": stds[0].unsqueeze(0),
        }

        img_original = img.clone()
        if self.color_mix["type"] == "source":
            if np.random.rand() < self.color_mix["freq"]:
                img = self.intensity_normalization(img_original, gt_semantic_seg, means, stds)

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True, mode="source"
        )
        src_feat = clean_losses.pop("features")
        seg_debug["Source"] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [p.grad.detach().clone() for p in params if p.grad is not None]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f"Seg. Grad.: {grad_mag}", "mmseg")

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)
            log_vars.update(add_prefix(feat_log, "src"))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [p.grad.detach() for p in params if p.grad is not None]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f"Fdist Grad.: {grad_mag}", "mmseg")
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only and (self.local_iter >= self.burnin):
            # Generate pseudo-label
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas
            )
            seg_debug["Target"] = self.get_ema_model().debug_output

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(ema_logits)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask
            )
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()

            mix_masks = get_class_masks(gt_semantic_seg)

            if self.color_mix["type"] != "none":
                self.contrast_flip.update(
                    img_original,
                    target_img,
                    gt_semantic_seg,
                    pseudo_label,
                    pseudo_weight,
                    strong_parameters,
                )

            if self.color_mix["type"] == "mix":
                img_color = self.contrast_flip.color_mix(
                    img, gt_semantic_seg, strong_parameters
                )
            else:
                img_color = img

            if self.mix == "class":
                for i in range(batch_size):
                    strong_parameters["mix"] = mix_masks[i]
                    mixed_img[i], mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((img_color[i], target_img[i])),
                        target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])),
                    )
                    _, mixed_seg_weight[i] = strong_transform(
                        strong_parameters,
                        target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])),
                    )

                del gt_pixel_weight
                mixed_img = torch.cat(mixed_img)
                mixed_lbl = torch.cat(mixed_lbl)
            else:
                for i in range(batch_size):
                    strong_parameters["mix"] = None
                    mixed_img[i], mixed_lbl[i] = strong_transform(
                        strong_parameters,
                        data=torch.stack((target_img[i],)),
                        target=torch.stack((pseudo_label[i].unsqueeze(0),)),
                    )
                    _, mixed_seg_weight[i] = strong_transform(
                        strong_parameters, target=torch.stack((pseudo_weight[i],))
                    )

                del gt_pixel_weight
                mixed_img = torch.cat(mixed_img)
                mixed_lbl = torch.cat(mixed_lbl)

            mixed_model = self.get_model()
            training_flag = True

            # for name, m in mixed_model.named_modules():
            #     if "normalization_net" in name:
            #         training_flag = False

            for name, m in mixed_model.named_modules():
                if "normalization_net" in name:
                    m.training = False
                else:
                    m.training = True

                # Train on mixed images

            mix_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
                mode="teacher",
            )

            seg_debug["Mix"] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, "mix")
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()

            # Contrastive loss
            if self.enable_contrastive and (self.local_iter >= self.burnin):
                contrast_losses = self.contrastive(
                    self.get_model(),
                    img,
                    gt_semantic_seg,
                    mixed_img,
                    mixed_lbl,
                    mixed_seg_weight,
                )

                seg_debug.update(self.contrastive.debug_output)
                contrast_losses = add_prefix(contrast_losses, "contrast")
                contrast_loss, contrast_log_vars = self._parse_losses(contrast_losses)
                log_vars.update(contrast_log_vars)
                contrast_loss.backward()

            for name, m in self.get_model().named_modules():
                m.training = training_flag

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith("separate"):
            masked_loss = self.mic(
                self.get_model(),
                img,
                img_metas,
                gt_semantic_seg,
                target_img,
                target_img_metas,
                valid_pseudo_mask,
                pseudo_label,
                pseudo_weight,
            )
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, "masked")
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        if (
            self.local_iter % self.debug_img_interval == 0
            and not self.source_only
            and (self.local_iter > self.burnin)
        ):
            out_dir = os.path.join(self.train_cfg["work_dir"], "debug")
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        "hspace": 0.1,
                        "wspace": 0,
                        "top": 0.95,
                        "bottom": 0,
                        "right": 1,
                        "left": 0,
                    },
                )
                subplotimg(axs[0][0], vis_img[j], "Source Image")
                subplotimg(axs[1][0], vis_trg_img[j], "Target Image")
                subplotimg(
                    axs[0][1], gt_semantic_seg[j], "Source Seg GT", cmap="cityscapes"
                )
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    "Target Seg (Pseudo) GT",
                    cmap="cityscapes",
                )
                subplotimg(axs[0][2], vis_mixed_img[j], "Mixed Image")
                subplotimg(axs[1][2], mix_masks[j][0], "Domain Mask", cmap="gray")
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                if mixed_lbl is not None:
                    subplotimg(axs[1][3], mixed_lbl[j], "Seg Targ", cmap="cityscapes")
                subplotimg(axs[0][3], mixed_seg_weight[j], "Pseudo W.", vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        "FDist Mask",
                        cmap="gray",
                    )
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        "Scaled GT",
                        cmap="cityscapes",
                    )
                for ax in axs.flat:
                    ax.axis("off")
                plt.savefig(
                    os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}.png")
                )
                plt.close()

        WandbLogImages(seg_debug)

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg["work_dir"], "debug")
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug["Source"] is not None and seg_debug:
                if "Target" in seg_debug:
                    seg_debug["Target"]["Pseudo W."] = mixed_seg_weight.cpu().numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={
                            "hspace": 0.1,
                            "wspace": 0,
                            "top": 0.95,
                            "bottom": 0,
                            "right": 1,
                            "left": 0,
                        },
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k2][k1],
                                **prepare_debug_out(
                                    f"{n1} {n2}", out[j], means, stds, n2
                                ),
                            )
                    for ax in axs.flat:
                        ax.axis("off")
                    plt.savefig(
                        os.path.join(out_dir, f"{(self.local_iter + 1):06d}_{j}_s.png")
                    )
                    plt.close()
                del seg_debug
        self.local_iter += 1

        return log_vars
