# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class RBFActivation(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super(RBFActivation, self).__init__()
        self.scale = nn.Parameter(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2) / (self.scale**2))


class NormNet(nn.Module):
    def __init__(self, norm_activation: str = "sigmoid", layers: List[int] = [1, 1]):
        super(NormNet, self).__init__()
        print("".join(["-"] * 80))
        print("Normalization network")
        print("NormNet: norm_activation =", norm_activation, "layers =", layers)

        norm_layers = []
        if norm_activation == "sigmoid":
            norm_layers.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
            norm_layers.append(nn.Sigmoid())
            norm_layers.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
        elif norm_activation == "linear":
            norm_layers.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
        else:
            for layer in range(1, len(layers)):
                norm_layers.append(
                    nn.Conv2d(
                        layers[layer - 1], layers[layer], kernel_size=1, bias=True
                    )
                )
                if layers[layer] != 1:
                    if norm_activation == "rbf":
                        init_value = torch.randn(layers[layer], 1, 1) * 0.05 + 0.2
                        norm_layers.append(RBFActivation(init_value))
                    elif norm_activation == "sine":
                        norm_layers.append(Sine())
                    elif norm_activation == "relu":
                        norm_layers.append(nn.ReLU())
                    else:
                        raise NotImplementedError("Activation function not implemented")

        self.norm_layers = nn.Sequential(*norm_layers)

        print(self.norm_layers)
        print("".join(["-"] * 80))

        # self.initialize_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.norm_layers(images)

    def initialize_weights(self) -> None:
        for m in self.norm_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
