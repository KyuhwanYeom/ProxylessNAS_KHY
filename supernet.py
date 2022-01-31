import torch.nn as nn
import time
from utils import *
from mixed_op import *


class MobileInvertedResidualBlock(nn.Module):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut is False:
            skip = 0
        else:
            skip = x
        return self.mobile_inverted_conv(x) + skip


class Supernets(nn.Module):

    def __init__(self, output_channels, conv_candidates,
                 n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0, n_cell = 3):
        super(Supernets, self).__init__()
        self._redundant_modules = None
        self._unused_modules = None
        self.label_smoothing = True

        # first conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU6()
        )

        # blocks
        first_cell_width = make_divisible(8 * width_mult, 8)
        input_channel = first_cell_width

        # blocks
        self.blocks = nn.ModuleList()
        for output_channel in output_channels:
            for i in range(n_cell):
                if i == 0:
                    stride = 2
                else:
                    stride = 1
                # conv
                if stride == 1 and input_channel == output_channel:
                    modified_conv_candidates = conv_candidates + ['Zero']
                    self.shortcut = True
                else:
                    modified_conv_candidates = conv_candidates
                    self.shortcut = False
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, output_channel, stride, 'weight_bn_act',
                ), )

                inverted_residual_block = MobileInvertedResidualBlock(conv_op, self.shortcut)
                self.blocks.append(inverted_residual_block)
                input_channel = output_channel

        # feature mix layer
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(100, 400, 1),
            nn.BatchNorm2d(400),
            nn.ReLU6(inplace=True)
        )
        # average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Linear Classifier
        self.classifier = nn.Linear(400, 10)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def init_weight(self, model_init):
        for m in self.modules():  # m은 각종 layer (Conv, BatchNorm, Linear ...)
            if isinstance(m, nn.Conv2d):  # conv layer면
                if model_init == 'he_fout':  # He initialization?
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):  # batch norm이면 weight = 1, bias = 0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):  # linear면 weight = uniform, bias = 0
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def reset_binary_gates(self):
        for m in self.redundant_modules:
            m.binarize()

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            involved_index = m.active_index + m.inactive_index
            for i in range(m.n_choices):  # n_choices : candiate path의 개수
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

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            m.set_arch_param_grad()

    def set_chosen_op_active(self):  # validate할 때 쓰임
        for m in self.redundant_modules:
            m.set_chosen_op_active() 

    def rescale_updated_arch_param(self):  # mix_op.py 238번째 줄
        for m in self.redundant_modules:
            m.rescale_updated_arch_param()
