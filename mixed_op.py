import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

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


def build_candidate_ops(candidate_ops, in_channels, out_channels, stride, ops_order):
    return [
        OPS[name](in_channels, out_channels, stride) for name in candidate_ops
    ]


class MixedEdge(nn.Module):
    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        self.AP_path_alpha = Parameter(torch.Tensor(
            self.n_choices))  # architecture parameters
        self.AP_path_wb = Parameter(
            torch.Tensor(self.n_choices))  # binary gates

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
    def random_op(self):
        index = np.random.choice([_i for _i in range(self.n_choices)], 1)[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = - torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        """ assume only one path is active """
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):  # validate 할 때 쓰임 (맨마지막!!!!!)
        chosen_idx, _ = self.chosen_index
        # inactive_index는 active index 제외 모두 (validate 이전의 초기화 부분에서는 inactive_index도 단 한개!)
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(
                                  chosen_idx + 1, self.n_choices)]

    """ """

    def forward(self, x):  # 여기가 forward!
        output = 0
        for _i in self.active_index:
            oi = self.candidate_ops[_i](x)
            output = output + self.AP_path_wb[_i] * oi
        for _i in self.inactive_index:
            oi = self.candidate_ops[_i](x)
            output = output + self.AP_path_wb[_i] * oi.detach()
        return output

    @property
    def module_str(self):
        chosen_index, probs = self.chosen_index
        return 'Mix(%s, %.3f)' % (self.candidate_ops[chosen_index].module_str, probs)

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        """ Only active paths taken into consideration when calculating FLOPs """
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    """ """

    def binarize(self):
        """ prepare: active_index, inactive_index, AP_path_wb, log_prob (optional), current_prob_over_ops (optional) """
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = F.softmax(self.AP_path_alpha, dim=0)
        if MixedEdge.MODE == 'two':
            # sample two ops according to `probs`
            sample_op = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in sample_op
            ]), dim=0)
            # chose one to be active and the other to be inactive according to probs_slice
            c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
            active_op = sample_op[c].item()
            inactive_op = sample_op[1 - c].item()
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary gate
            self.AP_path_wb.data[active_op] = 1.0

    def set_arch_param_grad(self):
        # ∂L/∂g (모든 path에 대한 gardient 구함)
        binary_grads = self.AP_path_wb.grad.data
        if self.active_op.is_zero_layer():
            self.AP_path_alpha.grad = None
            return
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if MixedEdge.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * \
                        (delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                # ex) self.active_index[0] = (3, 0.14301)
                self.active_index[_i] = (
                    idx, self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                # ex) self.inactive_index[0] = (1, 0.2313)
                self.inactive_index[_i] = (
                    idx, self.AP_path_alpha.data[idx].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[i] += binary_grads[j] * \
                        probs[j] * (delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        # ex) self.active_index[0] = (3, 0.14301)
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_idx = [idx for idx, _ in (
            self.active_index + self.inactive_index)]  # ex) [3, 1]
        # ex) [0.14301, 0.2313]
        old_alphas = [alpha for _, alpha in (
            self.active_index + self.inactive_index)]

        # ex) [tensor(0.1400), tensor(0.1200)] when AP_path_alpha = Parameter(torch.Tensor([0.1, 0.12, 0.13, 0.14]))
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) /
            sum([math.exp(alpha) for alpha in old_alphas])
        )

        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset
