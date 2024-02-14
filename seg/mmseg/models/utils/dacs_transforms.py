# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn
from mmseg.models.utils.dacs_normalization import NormNet
from torch import nn, optim

# import torchvision.transforms.functional as F
# import cv2 as cv
# import ot
# from scipy.optimize import linear_sum_assignment
# from skimage.exposure import match_histograms
# from torchvision.transforms.functional import equalize

class ClasswiseMultAugmenter:
    def __init__(self, n_classes, norm_type: str, suppress_bg: bool=True, device: str="cuda:0"):
        self.n_classes = n_classes
        self.device = device
        self.source_mean = -torch.ones(n_classes, 1)
        self.target_mean = -torch.ones(n_classes, 1)
        self.coef = torch.zeros(n_classes, 3)

        self.kernel_size = 3
        self.sigma = 0.5
        self.lambda_reg = 0.1
        self.suppress_bg = suppress_bg
        # self.matching_method = "sinkhorn"

        # self.learning_rate = 6e-05
        # self.learning_rate = 0.01
        # self.normalization_net = NormNet(norm_activation='rbf', cnn_layers = [1, 1]).to(device)
        # self.criterion = nn.MSELoss()
        # self.optimizer = optim.Adam(self.normalization_net.parameters(), lr=self.learning_rate, weight_decay=0.01)

    # def optimization_step(self, img_original, img_segm_hist, gt_semantic_seg):               
    #     self.optimizer.zero_grad()
    #     # print(img_original.device, img_segm_hist.device, gt_semantic_seg.device)
    #     # print(next(self.normalization_net.parameters()).device)
    #     # quit()

    #     img_polished = self.normalization_net(img_original[:, 0, :, :].unsqueeze(1)) 
        
    #     if self.suppress_bg:
    #          ## automatically detect background value
    #         # background_val = img_original[0, 0, 0, 0].item()
    #         # foreground_mask = img_original[:, 0, :, :].unsqueeze(1) > 0
    #         # background_mask = img_original[:, 0, :, :].unsqueeze(1) == background_val
            
    #         foreground_mask = gt_semantic_seg > 0
    #         background_mask = gt_semantic_seg == 0

    #         loss = self.criterion(img_polished[foreground_mask], img_segm_hist[foreground_mask].to(img_polished.device))
    #     else:
    #         loss = self.criterion(img_polished, img_segm_hist.to(img_polished.device))       

    #     loss.backward()
    #     self.optimizer.step()

    #     min_hist, max_hist = img_segm_hist[foreground_mask].min().item(), img_segm_hist[foreground_mask].max().item()

    #     # img = img_polished.detach()
    #     del img_polished
    #     with torch.no_grad():
    #         img = self.normalization_net(img_original[:, 0, :, :].unsqueeze(1)).detach()
            
    #     # img[foreground_mask] += max_hist - img[foreground_mask].max().item()

    #     if self.suppress_bg:
    #         img[background_mask] = img_segm_hist[background_mask]
    #         # img[background_mask] = img_original[:, 0, :, :].unsqueeze(1)[background_mask].mean().item()

    #     img = img.repeat(1, 3, 1, 1)

    #     return img, loss.item()
        
    def update(self, source, target, mask_src, mask_tgt, weight_tgt, param):
        mean, std = param["mean"], param["std"]
        denorm_(source, mean, std)
        denorm_(target, mean, std)

        c = 0
        for i in range(self.n_classes):
            source_mean = source[:, c, :, :][mask_src.squeeze(1) == i]
            target_mean = target[:, c, :, :][(mask_tgt.squeeze(1) == i) & (weight_tgt.squeeze(1) > 0)]

            if (source_mean.shape[0] != 0):
                self.source_mean[i, c] = source_mean.mean().item()

            if (target_mean.shape[0] != 0):
                self.target_mean[i, c] = target_mean.mean().item()

            if (self.source_mean[i, c] != -1) and (self.target_mean[i, c] != -1):
                self.coef[i, c] = self.target_mean[i, c] - self.source_mean[i, c]
                    
        renorm_(source, mean, std)
        renorm_(target, mean, std)

    def color_mix(self, data, mask, mean, std):
        data_ = data.clone()

        denorm_(data_, mean, std)

        for i in range(self.n_classes):
            for c in range(3):
                old_min = data_[:, c, :, :][mask.squeeze(1) != 0].min()
                data_[:, c, :, :][mask.squeeze(1) == i] += self.coef[i, 0]
                new_min = data_[:, c, :, :][mask.squeeze(1) != 0].min()

                data_[:, c, :, :][mask.squeeze(1) != 0] += old_min - new_min
            
        min_val = data_.min()
        data_ -= min_val
        max_val = data_.max()
        if max_val != 0:
            data_ /= max_val

        # min_val = data_.min()
        # max_val = data_.max()
        # data_ -= min_val
        # if max_val - min_val != 0:
        #     data_ /= (max_val - min_val)

        # for i in range(data_.shape[0]):
        #     gray_scale_img = torch.Tensor(match_histograms(data_[i, 0].cpu().numpy(), img_tgt_[i, 0].cpu().numpy(), channel_axis=-1))
        #     for c in range(3):
        #         data_[i, c] = gray_scale_img

        # for i in range(data_.shape[0]):
        #     for c in range(3):
        #         data_[i, c] = self._hungarian_match_cost(
        #             data_input[i, c].cpu().numpy(), data_[i, c].cpu().numpy(), 
        #             img_tgt[i, c].cpu().numpy()
        #         ).to(data_[i, c].device)

        renorm_(data_, mean, std)

        # return data_[:, 0, :, :].unsqueeze(1)
        return (data_[:, 0, :, :]).unsqueeze(1)

    # def _hungarian_match_cost(self, source, template, img_tgt):
    #     source = cv.blur(source, (2, 2))

    #     img_8b = np.array(source * 255).astype(np.uint8)

    #     template = match_histograms(template, img_tgt, channel_axis=-1)
    #     template = cv.blur(template, (3, 3))
    #     template_8b = np.array(template * 255).astype(np.uint8)
        
    #     npixels = 256
    #     cost_map = np.zeros((npixels, npixels))
    #     for i in range(img_8b.shape[0]):
    #         for j in range(img_8b.shape[1]):
    #             c1 = img_8b[i, j]
    #             c2 = template_8b[i, j]
    #             if c1 != 0 and c2 != 0:
    #                 cost_map[c1, c2] += 1
    #     cost_map /= cost_map.max()
    #     cost_map = 1 - cost_map
    #     cost_map[0, 0] = 0

    #     cost_map = cv.blur(cost_map, (2, 2))

    #     if self.matching_method == "sinkhorn":
    #         transport_matrix = ot.sinkhorn(
    #             np.ones(cost_map.shape[0]),
    #             np.ones(cost_map.shape[1]),
    #             cost_map,
    #             self.lambda_reg,
    #         )
    #         row_ind, col_ind = np.unravel_index(
    #             np.argmax(transport_matrix, axis=1), transport_matrix.shape
    #         )
    #     elif self.matching_method == "hungarian":
    #         row_ind, col_ind = linear_sum_assignment(cost_map)
    #     else:
    #         raise NotImplementedError
        
    #     img_8b_new = np.zeros(img_8b.shape)
    #     for i in range(img_8b.shape[0]):
    #         for j in range(img_8b.shape[1]):
    #             src_val = img_8b[i, j]
    #             img_8b_new[i, j] = col_ind[src_val]

    #     img_8b_new /= 255

    #     return torch.Tensor(img_8b_new)


def strong_transform(param, data=None, target=None):
    assert (data is not None) or (target is not None)

    data, target = one_mix(mask=param["mix"], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param["color_jitter"],
        s=param["color_jitter_s"],
        p=param["color_jitter_p"],
        mean=param["mean"],
        std=param["std"],
        data=data,
        target=target,
    )
    data, target = gaussian_blur(blur=param["blur"], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]["img_norm_cfg"]["mean"], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]["img_norm_cfg"]["std"], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0

def renorm(img, mean, std):
    return img.mul_(255.0).sub_(mean).div_(std)

def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=0.25, p=0.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s
                        )
                    )
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2])
                        - 0.5
                        + np.ceil(0.1 * data.shape[2]) % 2
                    )
                )
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3])
                        - 0.5
                        + np.ceil(0.1 * data.shape[3]) % 2
                    )
                )
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)
                    )
                )
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False
        )
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label, classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(
            0
        )
    return data, target
