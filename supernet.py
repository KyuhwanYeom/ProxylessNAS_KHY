import torch.nn as nn
from utils import *
from mixed_op import *


class MobileInvertedResidualBlock(nn.Module):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut is None:
            skip = 0
        else:
            skip = x
        return self.mobile_inverted_conv(x) + skip

class Supernets(nn.Module):

    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0):
        self._redundant_modules = None
        self._unused_modules = None

        input_channel = make_divisible(32 * width_mult, 8)
        first_cell_width = make_divisible(16 * width_mult, 8)
        for i in range(len(width_stages)):
            width_stages[i] = make_divisible(width_stages[i] * width_mult, 8)

        # first conv layer
        first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6()
        )

        #blocks
        first_block_conv = MixedEdge(OPS[2], input_channel, first_cell_width, 1, 'weight_bn_act')
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # feature mix layer
        feature_mix_layer = nn.Sequential(
            nn.Conv2d(3, 1280, 3, stride=2),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )

        # Linear Classifier
        classifier = nn.Linear(1280, 10)

