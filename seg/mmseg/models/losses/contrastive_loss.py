import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_entropy_loss import binary_cross_entropy, cross_entropy, mask_cross_entropy
from ..builder import LOSSES


class PixelContrastLoss(nn.Module):
    def __init__(
        self,
        temperature=0.1,
        base_temperature=0.1,
        ignore_label=-1,
        max_samples=1000000,
        max_views=8,
    ):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignore_label

        self.max_samples = max_samples
        self.max_views = max_views

    def _hard_anchor_sampling(self, X, y_hat, y, weight=None):
        batch_size, feat_dim = X.shape[0], X.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii] if weight is None else y_hat[ii][weight[ii] > 0]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            this_classes = [
                x
                for x in this_classes
                if (this_y == x).nonzero().shape[0] > self.max_views
            ]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            if weight is not None:
                this_w = weight[ii]

            for cls_id in this_classes:
                if weight is not None:
                    hard_indices = (
                        (this_y_hat == cls_id) & (this_y != cls_id) & (this_w > 0)
                    ).nonzero()
                    easy_indices = (
                        (this_y_hat == cls_id) & (this_y == cls_id) & (this_w > 0)
                    ).nonzero()

                else:
                    hard_indices = (
                        (this_y_hat == cls_id) & (this_y != cls_id)
                    ).nonzero()
                    easy_indices = (
                        (this_y_hat == cls_id) & (this_y == cls_id)
                    ).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print(
                        "this shoud be never touched! {} {} {}".format(
                            num_hard, num_easy, n_view
                        )
                    )
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0
        )
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def _correct_dim(self, labels, tgt_shape):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, tgt_shape, mode="nearest")

        return labels

    def forward(self, feats, labels, predict, weight=None):
        tgt_shape = (feats.shape[2], feats.shape[3])
        labels = self._correct_dim(labels, tgt_shape)
        labels = labels.squeeze(1).long()
        predict = self._correct_dim(predict, tgt_shape)
        predict = predict.squeeze(1).long()

        assert labels.shape[-1] == feats.shape[-1], "{} {}".format(
            labels.shape, feats.shape
        )
        assert predict.shape[-1] == feats.shape[-1], "{} {}".format(
            predict.shape, feats.shape
        )

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)

        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])

        if weight is not None:
            weight = self._correct_dim(weight, tgt_shape)
            weight = weight.contiguous().view(batch_size, -1)
            feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict, weight)
        else:
            feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


@LOSSES.register_module()
class ContrastCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        temperature=0.1,
        base_temperature=0.1,
        ignore_label=-1,
        max_samples=1000000,
        max_views=8,
    ):
        super(ContrastCELoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.class_weight = get_class_weight(class_weight)
        self.class_weight = None

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.pixel_contrast = PixelContrastLoss(
            temperature=temperature,
            base_temperature=base_temperature,
            ignore_label=ignore_label,
            max_samples=max_samples,
            max_views=max_views
        )

        self.debug = False
        self.debug_output = None

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        features=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        """Forward function."""

        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )

        if features is not None:
            _, predict = torch.max(cls_score.detach().clone(), 1)
            loss_cls += self.loss_weight * self.pixel_contrast(
                features, label, predict, weight
            )

        if self.debug:
            self.debug_output = {
                "Seg. Pred.": cls_score.detach().cpu().numpy(),
                "Seg. GT": label.detach().cpu().numpy(),
            }

        return loss_cls
    

@LOSSES.register_module()
class ContrastMixCELoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        use_sigmoid=False,
        use_mask=False,
        reduction="mean",
        class_weight=None,
        loss_weight=1.0,
        temperature=0.1,
        base_temperature=0.1,
        ignore_label=-1,
        max_samples=1000000,
        max_views=8,
    ):
        super(ContrastMixCELoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        # self.class_weight = get_class_weight(class_weight)
        self.class_weight = None

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        self.pixel_contrast = PixelContrastLoss(
            temperature=temperature,
            base_temperature=base_temperature,
            ignore_label=ignore_label,
            max_samples=max_samples,
            max_views=max_views
        )

        self.debug = False
        self.debug_output = None

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        features=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs
    ):
        """Forward function."""

        assert reduction_override in (None, "none", "mean", "sum")

        _, predict = torch.max(cls_score.detach().clone(), 1)

        if features is not None:
            loss_cls = self.loss_weight * self.pixel_contrast(
                features, label, predict, weight
            )
        else:
            raise NotImplementedError

        if self.debug:
            self.debug_output = {
                "Seg. Pred.": cls_score.detach().cpu().numpy(),
                "Seg. GT": label.detach().cpu().numpy(),
            }

        return loss_cls


if __name__ == "__main__":
    # torch.Size([1, 15, 256, 256]) torch.Size([1, 256, 256])
    num_classes = 3
    batch_size = 4
    logits1 = torch.randn(batch_size, 128, 256, 256)
    logits1 = F.normalize(logits1, dim=1)
    gt = torch.randint(low=0, high=num_classes, size=(batch_size, 256, 256))
    print(gt.unique())
    pred = torch.randn(batch_size, num_classes, 256, 256)
    # pred[0, :64, :64] = gt[0, :64, :64]
    # torch.randint(low=-1, high=15, size=(batch_size, 256, 256))

    loss = ContrastCELoss()

    loss1 = loss(logits1, pred, gt)
    print(loss1)
