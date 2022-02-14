import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import math
import gc
import os
import argparse

from utils import *
from supernet import *
from train_model import *
from torch.utils.data import random_split

# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {},
    'gpu8': {},
}

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

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


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

######################################################################
# Load model
print('==> Load model..')
assert os.path.isdir('output'), 'Error: no final directory found!'
Trained_model = SuperNets(
    output_channels=[32, 64, 128],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
)
final = torch.load('./output/final.pth')
print(final['net'])
# Trained_model.load_state_dict(final['net'])
# print(Trained_model.modules())
# optimizer_weight = optim.SGD(
#    Trained_model.weight_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
# Model_train(Trained_model, trainloader, validloader,
#            testloader, optimizer_weight)
