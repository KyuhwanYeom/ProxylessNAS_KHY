import torch.nn as nn
from collections import OrderedDict


class MBInvertedConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, expand_ratio):
        super(MBInvertedConvLayer, self).__init__()

        feature_dim = round(in_channels * expand_ratio)
        pad = kernel_size // 2
        self.expand_ratio = expand_ratio

        if expand_ratio != 1:
            self.inverted_bottleneck = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU6(inplace=True),
            )

        self.depth_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size,
                      stride, pad, groups=feature_dim, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU6(inplace=True),
        )

        self.point_linear = nn.Sequential(
            nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.expand_ratio != 1:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)
