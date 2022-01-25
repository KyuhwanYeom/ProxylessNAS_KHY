import torch
import torchvision
import torchvision.transforms as transforms
from supernet import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

super_net = Supernets( # over-parameterized net 생성 (큰 net)
    width_stages='24,40,80,96,192,320', n_cell_stages='4,4,4,4,4,1', stride_stages='2,2,2,1,2,1',
    conv_candidates=[
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ], n_classes=1000, width_mult=1.0,
    bn_param=(0.1, 1e-3), dropout_rate=0
)

