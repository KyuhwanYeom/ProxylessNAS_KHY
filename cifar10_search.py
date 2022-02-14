import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import math
import os
import argparse

from utils import *
from supernet import *
from train import *
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

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Search')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--gpu_devices", type=int,
                    nargs='+', default=None, help="")

args = parser.parse_args()

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

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

is_warmup = True
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

super_net = SuperNets(  # over-parameterized net 생성 (큰 net)
    output_channels=[32, 64, 128],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
)
print(super_net)
start = time.time()

# weight optimizer 정의 (momentum-SGD)
optimizer_weight = optim.SGD(
    super_net.weight_params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
# architecture parameter optimizer 정의 (Adam)
optimizer_arch = optim.Adam(
    super_net.arch_params, lr=0.006, weight_decay=5e-4)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('output'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./output/checkpoint.pth')
    super_net.load_state_dict(checkpoint['net'], strict=False)
    is_warmup = checkpoint['warmup']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

Trained = train(super_net, trainloader, validloader,
                testloader, optimizer_weight, optimizer_arch, is_warmup, best_acc, start_epoch)

end = time.time()

print(f"{end - start:.5f} sec")
