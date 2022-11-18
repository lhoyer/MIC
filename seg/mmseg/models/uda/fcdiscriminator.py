# Obtained from: https://github.com/wasidennis/AdaptSegNet
# Note from https://github.com/wasidennis/AdaptSegNet#note:
# The model and code are available for non-commercial research purposes only.

from torch import nn


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        conv_args = dict(kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(num_classes, ndf, **conv_args)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, **conv_args)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, **conv_args)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, **conv_args)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.classifier = nn.Conv2d(ndf * 8, 1, **conv_args)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x
