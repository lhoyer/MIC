# The adversarial domain adaptation is based on:
# https://github.com/wasidennis/AdaptSegNet
# Note from https://github.com/wasidennis/AdaptSegNet#note:
# The model and code are available for non-commercial research purposes only.

import os

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder
from mmseg.models.uda.fcdiscriminator import FCDiscriminator
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator
from mmseg.models.utils.dacs_transforms import denorm, get_mean_std
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.ops import resize


@UDA.register_module()
class AdvSeg(UDADecorator):

    def __init__(self, **cfg):
        super(AdvSeg, self).__init__(**cfg)
        self.local_iter = 0
        self.num_classes = cfg['model']['decode_head']['num_classes']
        self.max_iters = cfg['max_iters']
        self.lr_D = cfg['lr_D']
        self.lr_D_power = cfg['lr_D_power']
        self.lr_D_min = cfg['lr_D_min']
        self.discriminator_type = cfg['discriminator_type']
        self.lambda_adv_target = cfg['lambda_adv_target']
        self.mask_mode = cfg['mask_mode']

        self.model_D = nn.ModuleDict()
        self.optimizer_D = {}
        for k in ['main', 'aux'] if self.model.with_auxiliary_head \
                else ['main']:
            self.model_D[k] = FCDiscriminator(num_classes=self.num_classes)
            self.model_D[k].train()
            self.model_D[k].cuda()

            self.optimizer_D[k] = optim.Adam(
                self.model_D[k].parameters(), lr=self.lr_D, betas=(0.9, 0.99))
            self.optimizer_D[k].zero_grad()

        if self.discriminator_type == 'Vanilla':
            self.loss_fn_D = torch.nn.BCEWithLogitsLoss()
        elif self.discriminator_type == 'LS':
            self.loss_fn_D = torch.nn.MSELoss()
        else:
            raise NotImplementedError(self.discriminator_type)

        if self.mask_mode is not None:
            self.mic = MaskingConsistencyModule(require_teacher=True, cfg=cfg)

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
        for k in self.optimizer_D.keys():
            self.optimizer_D[k].zero_grad()
            self.adjust_learning_rate_D(self.optimizer_D[k], self.local_iter)
        log_vars = self(**data_batch)
        optimizer.step()
        for k in self.optimizer_D.keys():
            self.optimizer_D[k].step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def adjust_learning_rate_D(self, optimizer, i_iter):
        coeff = (1 - i_iter / self.max_iters)**self.lr_D_power
        lr = (self.lr_D - self.lr_D_min) * coeff + self.lr_D_min
        assert len(optimizer.param_groups) == 1
        optimizer.param_groups[0]['lr'] = lr

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
        source_label = 0
        target_label = 1

        self.update_debug_state()
        seg_debug = {}

        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        #######################################################################
        # Train Generator
        #######################################################################
        # don't accumulate grads in D
        for param in self.model_D.parameters():
            param.requires_grad = False

        # train with source
        source_losses = dict()
        pred = self.model.forward_with_aux(img, img_metas)
        loss = self.model.decode_head.losses(pred['main'], gt_semantic_seg)
        if self.get_model().debug:
            self.get_model().process_debug(img, img_metas)
            seg_debug['Source'] = self.get_model().debug_output
            self.get_model().debug_output = {}
        source_losses.update(add_prefix(loss, 'decode'))
        if isinstance(self.model, HRDAEncoderDecoder):
            self.model.decode_head.reset_crop()
        if self.model.with_auxiliary_head:
            loss_aux = self.model.auxiliary_head.losses(
                pred['aux'], gt_semantic_seg)
            source_losses.update(add_prefix(loss_aux, 'aux'))
        source_loss, source_log_vars = self._parse_losses(source_losses)
        source_loss.backward()

        # train with target
        pred_trg = self.model.forward_with_aux(target_img, target_img_metas)
        if isinstance(self.model, HRDAEncoderDecoder):
            self.model.decode_head.reset_crop()

        if isinstance(self.model, HRDAEncoderDecoder):
            for k in pred.keys():
                pred[k] = pred[k][0]
                assert self.model.feature_scale == 0.5
                pred[k] = resize(
                    input=pred[k],
                    size=[
                        int(e * self.model.feature_scale)
                        for e in img.shape[2:]
                    ],
                    mode='bilinear',
                    align_corners=self.model.align_corners)
            for k in pred_trg.keys():
                pred_trg[k] = pred_trg[k][0]
                pred_trg[k] = resize(
                    input=pred_trg[k],
                    size=[
                        int(e * self.model.feature_scale)
                        for e in img.shape[2:]
                    ],
                    mode='bilinear',
                    align_corners=self.model.align_corners)

        g_trg_losses = dict()
        for k in pred_trg.keys():
            D_out = self.model_D[k](F.softmax(pred_trg[k], dim=1))
            loss_G = self.loss_fn_D(
                D_out,
                Variable(
                    torch.FloatTensor(
                        D_out.data.size()).fill_(source_label)).cuda())
            # remember to have the word 'loss' in key
            g_trg_losses[
                f'G_trg.loss.{k}'] = self.lambda_adv_target[k] * loss_G
        g_trg_loss, g_trg_log_vars = self._parse_losses(g_trg_losses)
        g_trg_loss.backward()

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

        #######################################################################
        # Train Discriminator
        #######################################################################
        # bring back requires_grad
        for param in self.model_D.parameters():
            param.requires_grad = True

        # train with source
        d_src_losses = dict()
        for k in pred.keys():
            pred[k] = pred[k].detach()
            D_out_src = self.model_D[k](F.softmax(pred[k], dim=1))
            loss_D = self.loss_fn_D(
                D_out_src,
                Variable(
                    torch.FloatTensor(
                        D_out_src.data.size()).fill_(source_label)).cuda())
            d_src_losses[f'D_src.loss.{k}'] = loss_D / 2
        d_src_loss, d_src_log_vars = self._parse_losses(d_src_losses)
        d_src_loss.backward()

        # train with target
        d_trg_losses = dict()
        for k in pred_trg.keys():
            pred_trg[k] = pred_trg[k].detach()
            D_out_trg = self.model_D[k](F.softmax(pred_trg[k], dim=1))
            loss_D = self.loss_fn_D(
                D_out_trg,
                Variable(
                    torch.FloatTensor(
                        D_out_trg.data.size()).fill_(target_label)).cuda())
            d_trg_losses[f'D_trg.loss.{k}'] = loss_D / 2
        d_trg_loss, d_trg_log_vars = self._parse_losses(d_trg_losses)
        d_trg_loss.backward()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            batch_size = img.shape[0]
            means, stds = get_mean_std(img_metas, target_img.device)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
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
                subplotimg(
                    axs[0][1],
                    torch.argmax(pred['main'][j], dim=0),
                    'Source Seg',
                    cmap='cityscapes')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[1][1],
                    torch.argmax(pred_trg['main'][j], dim=0),
                    'Target Seg',
                    cmap='cityscapes')
                subplotimg(
                    axs[0][2],
                    D_out_src[j],
                    'Source Discriminator',
                    vmin=0,
                    vmax=1,
                    cmap='viridis')
                subplotimg(
                    axs[1][2],
                    D_out_trg[j],
                    'Target Discriminator',
                    vmin=0,
                    vmax=1,
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
        return {
            **source_log_vars,
            **g_trg_log_vars,
            **d_src_log_vars,
            **d_trg_log_vars,
            **masked_log_vars,
        }
