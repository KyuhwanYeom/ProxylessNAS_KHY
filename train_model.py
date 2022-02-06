import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
from torch.utils.tensorboard import SummaryWriter
from utils import *
from mixed_op import *


class Model_train():
    def __init__(self, net, trainloader, validloader, testloader, optimizer_weight):
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.optimizer_weight = optimizer_weight
        MixedEdge.MODE = 'None'
        self.writer = SummaryWriter()

        self.multiGPU()
        
        self.train()
        self.validate()
        self.test()

    def multiGPU(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # gpu 사용
        self.net.to(self.device)
        
    def train_one_epoch(self, adjust_lr_func):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.trainloader):
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)

            # compute output
            output = self.net(images)
            if self.net.module.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(
                    output, labels, 0.1)
            else:
                loss = nn.CrossEntropyLoss(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer_weight.step()
            progress_bar(i, len(self.trainloader), 'Train Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                             % (losses.avg, top1.avg, top5.avg))

        return top1, top5

    def train(self, train_epochs=300):
        nBatch = len(self.trainloader)
        for epoch in range(0, train_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' %
                  (epoch + 1), '-' * 30, '\n')
            train_top1, train_top5 = self.train_one_epoch(
                lambda i: self.adjust_learning_rate(
                    self.optimizer_weight, epoch, i, nBatch)
            )
            (val_loss, val_top1, val_top5) = self.validate()  # validation 진행


    # cosine annealing 사용
    def _calc_learning_rate(self, epoch, batch, nBatch, n_epochs=300):
        T_total = n_epochs * nBatch
        T_cur = epoch * nBatch + batch
        lr = 0.5 * 0.01 * (1 + math.cos(math.pi * T_cur / T_total))
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch, nBatch):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self._calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

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
        # test on validation set under train mode
        valid_res = self.validate_validloader()
        return valid_res

    def validate_validloader(self):
        self.net.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

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
                progress_bar(i, len(self.validloader), 'Valid Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                             % (losses.avg, top1.avg, top5.avg))

        return losses.avg, top1.avg, top5.avg
