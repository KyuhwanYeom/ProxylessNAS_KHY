import torch.nn as nn
import time
from normal_net import NormalNets
from utils import *
from mixed_op import *
from queue import Queue


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


class SuperNets(nn.Module):

    def __init__(self, output_channels, conv_candidates,
                 n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0, n_cell=3):
        super().__init__()
        self.redundant_modules = []
        self.unused_modules = []
        self.label_smoothing = True
        self.arch_params = []
        self.weight_params = []
        self.model_init = 'he fout'

        # first conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU6()
        )

        input_channel = 32

        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'], input_channel, 16, 1))
        first_block = MobileInvertedResidualBlock(first_block_conv, False)
        input_channel = 16

        # blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(first_block)

        for output_channel in output_channels:
            for i in range(n_cell):
                stride = 2 if i == 0 else 1

                # conv
                if stride == 1 and input_channel == output_channel:
                    modified_conv_candidates = conv_candidates + ['Zero']
                    self.shortcut = True
                else:
                    modified_conv_candidates = conv_candidates
                    self.shortcut = False
                conv_op = MixedEdge(build_candidate_ops(
                    modified_conv_candidates, input_channel, output_channel, stride))

                inverted_residual_block = MobileInvertedResidualBlock(
                    conv_op, self.shortcut)
                self.blocks.append(inverted_residual_block)
                input_channel = output_channel

        # feature_mix_layer
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(128, 1280, 1, stride=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )

        # average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Linear Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(inplace=True),
            nn.Linear(1280, 10)
        )

        # super(SuperNets, self).__init__(self.first_conv,
        #                                self.blocks, self.feature_mix_layer, self.classifier)

        self.init_modules()
        self.init_parameters()

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    def init_weight(self, model_init):
        for m in self.modules():  # m은 각종 layer (Conv, BatchNorm, Linear ...)
            if isinstance(m, nn.Conv2d):  # conv layer면 He initialization
                if model_init == 'he_fout':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif model_init == 'he_fin':
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            # batch norm이면 weight = 1, bias = 0
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):  # linear면 weight = uniform, bias = 0
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.zero_()

    def init_modules(self):
        for m in self.modules():
            if m.__str__().startswith('MixedEdge'):
                self.redundant_modules.append(m)

    def init_parameters(self):
        self.init_weight(self.model_init)
        for params in self.named_parameters():
            if 'AP_path_alpha' in params[0]:
                params[1].data.normal_(0, 1e+1)
                self.arch_params.append(params[1])
            else:
                self.weight_params.append(params[1])

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            m.binarize()

    def unused_modules_off(self):
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE == 'NORMAL':
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):  # n_choices : candiate path의 개수
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self.unused_modules.append(unused)

    def unused_modules_back(self):
        for m, unused in zip(self.redundant_modules, self.unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self.unused_modules = []

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            m.set_arch_param_grad()

    def set_chosen_op_active(self):  # validate할 때 쓰임
        for m in self.redundant_modules:
            m.set_chosen_op_active()

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            m.rescale_updated_arch_param()

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return Normalnets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)
