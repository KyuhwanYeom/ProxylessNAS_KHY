import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import math
import gc
import os
import argparse
import cifar10_search

from utils import *
from supernet import *
from train_searched_model import *
from torch.utils.data import random_split
from cifar10 import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--gpu_devices", type=int,
                    nargs='+', default=None, help="")
parser.add_argument('--n_cell_stages', type=list, default=[3, 3, 3])
parser.add_argument('--stride_stages', type=list, default=[1, 2, 2])
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')
args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

cifar10 = Cifar10DataProvider(cutout=args.cutout)

"""
# Data prepare
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


transform_valid = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_valid)


train_ds, val_ds = random_split(trainset, [45000, 5000])

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=256,  # 45000
                                          shuffle=True, num_workers=16)

validloader = torch.utils.data.DataLoader(val_ds, batch_size=256,  # 5000
                                          shuffle=True, num_workers=16)

testloader = torch.utils.data.DataLoader(testset, batch_size=256,  # 10000
                                         shuffle=False, num_workers=16)

"""

######################################################################
# Load model
print('==> Load model..')
assert os.path.isdir('output'), 'Error: no final directory found!'

Trained_model = SuperNets(  # over-parameterized net 생성 (큰 net)
    output_channels=[32, 64, 128],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ],
    stride_stages=args.stride_stages,
    n_cell_stages=args.n_cell_stages
).cuda()

final = torch.load('./output/final.pth')
Trained_model.load_state_dict(final['state_dict'], strict=False)
Trained_model = Trained_model.convert_to_normal_net()
Trained_model.init_weight(model_init="he_fout")
optimizer_weight = optim.SGD(Trained_model.weight_parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)
Model_train(Trained_model, cifar10.train, cifar10.valid,
            cifar10.test, optimizer_weight, args)
