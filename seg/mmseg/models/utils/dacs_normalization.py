# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import torch
import torch.nn as nn
import torch.nn.functional as F

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)

class RBFActivation(nn.Module):
    def __init__(self, scale):
        super(RBFActivation, self).__init__()
        self.scale = nn.Parameter(scale)
        
    def forward(self, x):
        return torch.exp(-(x ** 2) / (self.scale ** 2))
    
class NormNet(nn.Module):
    def __init__(self, norm_activation = 'sigmoid', cnn_layers = [1, 1]):
        super(NormNet, self).__init__()        
        print('NormNet: norm_activation =', norm_activation, 'cnn_layers =', cnn_layers)

        layers = []
        if norm_activation == 'sigmoid':
            layers.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
            layers.append(nn.Sigmoid())
            layers.append(nn.Conv2d(1, 1, kernel_size=1, bias=True))
        else:
            for layer in range(1, len(cnn_layers)):
                layers.append(nn.Conv2d(cnn_layers[layer-1], cnn_layers[layer], kernel_size=1, bias=True))
                if cnn_layers[layer] != 1:                    
                    if norm_activation == 'rbf':
                        init_value = torch.randn(cnn_layers[layer], 1, 1) * 0.05 + 0.2
                        layers.append(RBFActivation(init_value))
                    elif norm_activation == 'sine':
                        layers.append(Sine())
                    elif norm_activation == 'relu':
                        layers.append(nn.ReLU())
                    else:
                        raise NotImplementedError('Activation function not implemented')

        self.norm_layers = nn.Sequential(*layers)

        print('Normalization network')
        print(self.norm_layers)

        # self.initialize_weights()
        
    def forward(self, images):
        return self.norm_layers(images)
    
    def initialize_weights(self):
        for m in self.norm_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)