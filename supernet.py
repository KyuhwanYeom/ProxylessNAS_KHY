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
        super().__init__()
        self._redundant_modules = None
        self._unused_modules = None

        # first conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        #blocks
        input_channel = 32
        first_cell_width = make_divisible(16 * width_mult, 8)

        self.first_block_conv = MixedEdge(
            [OPS['3x3_MBConv1'](input_channel, first_cell_width, 1)])
        if self.first_block_conv.n_choices == 1:
            first_block_conv = self.first_block_conv.candidate_ops[0]
        first_block = MobileInvertedResidualBlock(first_block_conv, None)
        input_channel = first_cell_width

        # feature mix layer
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(3, 400, 3, stride=2),
            nn.BatchNorm2d(400),
            nn.ReLU6()
        )

        # Linear Classifier
        self.classifier = nn.Linear(400, 10)
    
    def reset_binary_gates(self):
        for m in self.modules():
            m.binarize()

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            involved_index = m.active_index + m.inactive_index
            for i in range(m.n_choices): # n_choices : candiate path의 개수
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)
    
    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None