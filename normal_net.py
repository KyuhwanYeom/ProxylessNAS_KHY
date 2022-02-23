import torch
import torch.nn as nn
import math

class NormalNets(nn.Module):

    def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
        super(NormalNets, self).__init__()
        self.weight_params = []
        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

                    
    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
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

    def weight_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param