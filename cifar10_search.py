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
from arch_search import *
from torch.utils.data import random_split
from cifar10 import *

if __name__ == '__main__':
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
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument("--gpu_devices", type=int,
                        nargs='+', default=None, help="")
    parser.add_argument('--n_cell_stages', type=list, default=[3, 3, 3])
    parser.add_argument('--stride_stages', type=list, default=[1, 2, 2])
    parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout')

    """ train setting """
    parser.add_argument('--warm_up_epoch', default=40,
                        type=int, help='warm up epoch')
    parser.add_argument('--lr', default=0.025,
                        type=float, help='learning rate')
    parser.add_argument('--train_epoch', default=120,
                        type=int, help='train epoch')

    """ regularization loss setting """
    parser.add_argument('--grad_reg_loss_type', type=str,
                        default='add#linear', choices=['add#linear', 'mul#log'])
    parser.add_argument('--grad_reg_loss_lambda', type=float,
                        default=1e-1)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_alpha', type=float,
                        default=0.2)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_beta', type=float,
                        default=0.3)  # grad_reg_loss_params
    parser.add_argument('--target_hardware', type=str,
                        default=None, choices=['flops', None])

    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    gpu_devices = ','.join([str(id) for id in args.gpu_devices])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

    data_cifar10 = Cifar10DataProvider(cutout=args.cutout)

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

    is_warmup = True
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    super_net = SuperNets(  # over-parameterized net 생성 (큰 net)
        output_channels=[32, 64, 128],
        conv_candidates=[
            '3x3_MBConv3', '3x3_MBConv6',
            '5x5_MBConv3', '5x5_MBConv6',
            '7x7_MBConv3', '7x7_MBConv6',
        ],
        stride_stages=args.stride_stages,
        n_cell_stages=args.n_cell_stages
    ).cuda()

    if args.target_hardware is None:
        args.ref_value = None
    else:
        args.ref_value = ref_values[args.target_hardware]['1.00']
    if args.grad_reg_loss_type == 'add#linear':
        args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == 'mul#log':
        args.grad_reg_loss_params = {
            'alpha': args.grad_reg_loss_alpha,
            'beta': args.grad_reg_loss_beta,
        }

    start = time.time()
    arch_search_run_manager = ArchSearch(
        super_net, data_cifar10, is_warmup, start_epoch, args)

    if arch_search_run_manager.is_warmup:
        lr = arch_search_run_manager.warm_up()
    arch_search_run_manager.train()

    end = time.time()

    print(f"{end - start:.5f} sec")
