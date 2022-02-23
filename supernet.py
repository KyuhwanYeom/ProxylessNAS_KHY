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
        if isinstance(self.mobile_inverted_conv, Zero):
            return x
        if self.shortcut is False:
            skip = 0
        else:
            skip = x
        return self.mobile_inverted_conv(x) + skip

    def get_flops(self, x):
        flops1, _ = self.mobile_inverted_conv.get_flops(x)

        return flops1, self.forward(x)


class SuperNets(nn.Module):

    def __init__(self, output_channels, conv_candidates, stride_stages, n_cell_stages,
                 n_classes=10, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0, n_cell=3):
        super().__init__()
        self._redundant_modules = None
        self._unused_modules = None
        self.label_smoothing = True
        self.model_init = 'he fout'

        self.stride_stages = stride_stages
        self.n_cell_stages = n_cell_stages
        input_channel = 32
        first_cell_width = 16

        # first conv layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6()
        )

        # first conv layer
        self.first_conv_copy = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6()
        )

        self.shortcut = False
        first_block_conv = MixedEdge(candidate_ops=build_candidate_ops(
            ['3x3_MBConv1'], input_channel, first_cell_width, 1))
        if first_block_conv.n_choices == 1:
            first_block_conv = first_block_conv.candidate_ops[0]

        first_block = MobileInvertedResidualBlock(
            first_block_conv, self.shortcut)
        input_channel = first_cell_width

        input_channel = first_cell_width

        # blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(first_block)

        for width, n_cell, s in zip(output_channels, self.n_cell_stages, self.stride_stages):
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # conv
                if stride == 1 and input_channel == width:
                    modified_conv_candidates = conv_candidates + ['Zero']
                    self.shortcut = True
                else:
                    modified_conv_candidates = conv_candidates
                    self.shortcut = False
                conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, input_channel, width, stride))

                inverted_residual_block = MobileInvertedResidualBlock(
                    conv_op, self.shortcut)
                self.blocks.append(inverted_residual_block)
                input_channel = width

        # feature_mix_layer
        self.feature_mix_layer = nn.Sequential(
            nn.Conv2d(128, 1280, 1, stride=1),
            nn.BatchNorm2d(1280, eps=1e-3),
            nn.ReLU6()
        )

        # average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Linear Classifier
        self.classifier = nn.Linear(1280, 10)

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

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self):
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def init_parameters(self):
        self.init_weight(self.model_init)
        for params in self.architecture_parameters():
            params.data.normal_(0, 1e-3)

    def reset_binary_gates(self):
        for m in self.redundant_modules:
            m.binarize()

    def print_active_index(self):
        for m in self.redundant_modules:
            print(m.probs_over_ops.data.cpu().numpy())

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE == None:
                involved_index = m.active_index
            else:
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
        return NormalNets(self.first_conv, list(self.blocks), self.feature_mix_layer, self.classifier)

    def get_flops(self, x):

        flop, x = count_conv_flop(self.first_conv[0], x), self.first_conv(x)

        for block in self.blocks:
            delta_flop, x = block.get_flops(x)
            flop += delta_flop

        delta_flop, x = count_conv_flop(
            self.feature_mix_layer[0], x), self.feature_mix_layer(x)
        flop += delta_flop

        x = self.gap(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier.weight.numel(), self.classifier(x)
        flop += delta_flop
        return flop, x

    def expected_flops(self, x):
        expected_flops = 0
        # first conv
        flop = count_conv_flop(self.first_conv[0], x)
        x = self.first_conv(x)
        expected_flops += flop

        # blocks
        for block in self.blocks:
            mb_conv = block.mobile_inverted_conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or isinstance(op, Zero):
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops = expected_flops + op_flops * probs_over_ops[i]
            x = block(x)

        # feature mix layer
        delta_flop = count_conv_flop(self.feature_mix_layer[0], x)
        x = self.feature_mix_layer(x)
        expected_flops = expected_flops + delta_flop
        # classifier
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop = self.classifier.weight.numel()
        x = self.classifier(x)
        expected_flops = expected_flops + delta_flop

        return expected_flops
