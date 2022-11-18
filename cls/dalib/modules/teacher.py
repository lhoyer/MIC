from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class EMATeacher(nn.Module):

    def __init__(self, model, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        logits, _ = self.ema_model(target_img)

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight
