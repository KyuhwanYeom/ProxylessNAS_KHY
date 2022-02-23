import numpy as np
import torch
import torch.nn.functional as F
import math

from torch.nn.parameter import Parameter
from layers import *

OPS = {
    'Identity': lambda in_C, out_C, S: Identity(in_C, out_C),
    'Zero': lambda in_C, out_C, S: Zero(stride=S),
    '3x3_MBConv1': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 1),
    '3x3_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 3),
    '3x3_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 3, S, 6),
    #######################################################################################
    '5x5_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 3),
    '5x5_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 5, S, 6),
    #######################################################################################
    '7x7_MBConv3': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 3),
    '7x7_MBConv6': lambda in_C, out_C, S: MBInvertedConvLayer(in_C, out_C, 7, S, 6),
}


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride):
    return [
        OPS[name](in_channels, out_channels, stride) for name in candidate_ops
    ]


class MixedEdge(nn.Module):
    MODE = None

    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))  # binary gates

        self.active_index = [0]
        self.inactive_index = None


    @property
    def n_choices(self):
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):  # probs는 배열이 됨
        probs = F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability
        return probs

    @property
    def chosen_index(self):
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):  # validate 할 때 쓰임 (맨마지막!!!!!)
        chosen_idx, _ = self.chosen_index
        # inactive_index는 active index 제외 모두 (validate 이전의 초기화 부분에서는 inactive_index도 단 한개!)
        self.active_index = [chosen_idx]
        self.inactive_index = [i for i in range(0, chosen_idx)] + \
                              [i for i in range(chosen_idx + 1, self.n_choices)]

    """ """

    def forward(self, x):  # 여기가 forward!
        if MixedEdge.MODE == None:
            output = self.active_op(x)
        else:
            output = 0
            output = output + self.candidate_ops[self.active_index[0]](x)
            for i in self.inactive_index:
                output = output + \
                    self.AP_path_wb[i] * self.candidate_ops[i](x).detach()

        return output

    """ """
    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)
    
    def binarize(self):
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        if MixedEdge.MODE == None:
            sample = torch.multinomial(probs.data, 1)[0].item()
            self.active_index = [sample]
            self.inactive_index = [i for i in range(0, sample)] + \
                                  [i for i in range(sample + 1, self.n_choices)]
            self.current_prob_over_ops = probs
            # set binary gate
            self.AP_path_wb.data[sample] = 1.0
        else:
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, idx in enumerate(sample_op):
                self.current_prob_over_ops[idx] = probs_slice[i]
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0
        # avoid over-regularization
        for i in range(self.n_choices):
            for name, param in self.candidate_ops[i].named_parameters():
                param.grad = None

    def delta_ij(self, i, j):
        if i == j:
            return 1
        else:
            return 0

    def set_arch_param_grad(self):
        # ∂L/∂g (모든 path에 대한 gardient 구함)
        binary_grads = self.AP_path_wb.grad.data
        if isinstance(self.active_op, Zero):
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
            
        involved_idx = self.active_index + self.inactive_index
        probs = F.softmax(torch.stack([
            self.AP_path_alpha[idx] for idx in involved_idx
        ]), dim=0).data
        for i in range(2):
            for j in range(2):
                origin_i = involved_idx[i]
                origin_j = involved_idx[j]
                self.AP_path_alpha.grad.data[origin_i] += \
                    binary_grads[origin_j] * probs[j] * \
                    (self.delta_ij(i, j) - probs[i])

    def rescale_updated_arch_param(self):        
        involved_idx = self.active_index + self.inactive_index  # ex) [3, 1]
        old_alphas = [self.AP_path_alpha.data[self.active_index].item(
        )] + [self.AP_path_alpha.data[self.inactive_index].item()]

        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) /
            sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset