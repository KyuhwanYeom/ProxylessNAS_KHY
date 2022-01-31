import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import math

from supernet import *
from train import *
from torch.utils.data import random_split

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_ds, val_ds = random_split(trainset, [45000, 5000])

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=256, # 45000
                                          shuffle=True, num_workers=32)

validloader = torch.utils.data.DataLoader(val_ds, batch_size=256, # 5000
                                          shuffle=True, num_workers=32)

testloader = torch.utils.data.DataLoader(testset, batch_size=256, # 10000
                                         shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
n_cell = 18
model_init = 'he_fout'

super_net = Supernets(  # over-parameterized net 생성 (큰 net)
    output_channels=[8, 20, 40, 64, 80, 100],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ], n_classes=10, width_mult=1,
    bn_param=(0.1, 1e-3), dropout_rate=0
)

super_net.init_weight(model_init) # weight initialization (He)

arch_params = []  # init architecture parameter ,weight parameter
weight_params = []
for x in super_net.named_parameters():
    if 'alpha' in x[0]:
        x[1].data.normal_(0, 1e-3) 
        arch_params.append(x[1])
    else:
        weight_params.append(x[1])

# weight optimizer 정의 (momentum-SGD)
optimizer_weight = optim.SGD(weight_params, lr=0.001, momentum=0.9)
# architecture parameter optimizer 정의 (Adam)
optimizer_arch = optim.Adam(arch_params, lr=1e-3)

train(super_net, trainloader, validloader ,testloader, optimizer_weight, optimizer_arch)

