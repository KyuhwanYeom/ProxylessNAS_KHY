import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import math
import gc

from supernet import *
from train import *
from train_model import *
from torch.utils.data import random_split

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_ds, val_ds = random_split(trainset, [45000, 5000])

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=512,  # 45000
                                          shuffle=True, num_workers=16)

validloader = torch.utils.data.DataLoader(val_ds, batch_size=512,  # 5000
                                          shuffle=True, num_workers=16)

testloader = torch.utils.data.DataLoader(testset, batch_size=512,  # 10000
                                         shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

super_net = Supernets(  # over-parameterized net 생성 (큰 net)
    output_channels=[32, 64, 128],
    #output_channels=[8, 20, 40, 64, 80, 100],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
)

start = time.time()

# weight optimizer 정의 (momentum-SGD)
optimizer_weight = optim.SGD(super_net.weight_params, lr=0.01, momentum=0.9)
# architecture parameter optimizer 정의 (Adam)
optimizer_arch = optim.Adam(super_net.arch_params, lr=0.006)

Trained = train(super_net, trainloader, validloader,
                testloader, optimizer_weight, optimizer_arch)

end = time.time()

Model_train(Trained.net, trainloader, validloader,
            testloader, Trained.optimizer_weight)
################################################
# train model

checkpoint = torch.load("./output/checkpoint.pth")
model = Supernets(
    output_channels=[32, 64, 128],
    #output_channels=[8, 20, 40, 64, 80, 100],
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
)
optimizer_weight = optim.SGD(super_net.weight_params, lr=0.05, momentum=0.9)
optimizer_arch = optim.Adam(super_net.arch_params, lr=0.006)

model.load_state_dict(checkpoint['state_dict'])
optimizer_weight.load_state_dict(checkpoint['weight_optimizer'])
optimizer_arch.load_state_dict(checkpoint['arch_optimizer'])

"""
Model_train(model, trainloader, validloader,
            testloader, optimizer_weight, optimizer_arch)
"""

print(f"{end - start:.5f} sec")
