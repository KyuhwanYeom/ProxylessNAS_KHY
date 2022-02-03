import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils import *


class Model_train():
    def __init__(self, net, trainloader, validloader, testloader, optimizer_weight, optimizer_alpha):
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.optimizer_weight = optimizer_weight
        self.optimizer_alpha = optimizer_alpha
        self.writer = SummaryWriter()

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu 사용
        self.net = nn.DataParallel(self.net)
        # print(device)
        self.net.to(self.device)
        self.net = self.net.module  # dataparallel
        self.train()
        self.validate()
        self.test()

    def train(self, train_epochs=50):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=train_epochs+1, T_mult=1, eta_min=0.00001)  # cosine annealing
        writer2 = SummaryWriter()
        for epoch in range(0, train_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' %
                  (epoch + 1), '-' * 30, '\n')
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.net.train()

            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data[0].to(
                    self.device), data[1].to(self.device)
                # compute output
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.label_smoothing:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, 0.1)
                else:
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                # compute gradient and do SGD step
                # zero grads of weight_param, arch_param & binary_param
                self.net.zero_grad()
                loss.backward()
                self.optimizer_weight.step()  # update weight parameters
                # unused modules back
                # write in tensorboard
                if i % 5 == 0:
                    # print(loss.item())
                    self.writer.add_scalar(
                        'train loss', loss.item(), epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('train_top-1 acc', top1.val,
                                           epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('train_top-5 acc', top5.val,
                                           epoch*len(self.trainloader) + i)
                    (val_loss, val_top1, val_top5) = self.validate()  # validation 진행
                    print(f'Validation Loss : {val_loss}')
                    print(f'Valid Top-1 acc : {val_top1}')
                    print(f'Valid Top-5 acc : {val_top5}')

            scheduler.step()
            print(f'Loss : {losses.val:.4f}, {losses.avg:.4f}')
            print(f'Top-1 acc : {top1.val:.3f}, {top1.avg:.3f}')
            print(f'Top-5 acc : {top5.val:.3f}, {top5.avg:.3f}')
            print(f'learning rate : {scheduler.get_lr()[0]}')

            self.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.optimizer_weight.state_dict(),
                'arch_optimizer': self.optimizer_alpha.state_dict(),
                'state_dict': self.net.state_dict(),
                'blocks': self.net.blocks
            })

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(
                    self.device), data[1].to(self.device)
                # 신경망에 이미지를 통과시켜 출력을 계산
                outputs = self.net(images)
                # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    def validate(self):
        # set chosen op active
        self.net.set_chosen_op_active()
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.validate_validloader()
        # unused modules back
        self.net.unused_modules_back()
        return valid_res

    def validate_validloader(self):
        self.net.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.validloader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = self.net(images)
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

        return losses.avg, top1.avg, top5.avg