import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils import *


class train():
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
        self.lr = self.warm_up()
        self.train(self.lr)
        self.test()

    def warm_up(self, warmup=0, warmup_epochs=1):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=warmup_epochs + 1, T_mult=1, eta_min=1e-3)  # cosine annealing
        for epoch in range(warmup, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' %
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
                self.net.reset_binary_gates()  # random sample binary gates
                # remove unused module for speedup
                self.net.unused_modules_off()
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.label_smoothing:
                    loss = cross_entropy_with_label_smoothing(
                        output, labels, 0.1)
                else:
                    loss = nn.CrossEntropyLoss(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                # compute gradient and do SGD step
                self.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()

                # write in tensorboard
                if i % 5 == 0:
                    # print(loss.item())
                    self.writer.add_scalar(
                        'warm_up loss', loss.item(), epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('warm-up_top-1 acc', top1.val,
                                           epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('warm-up_top-5 acc', top5.val,
                                           epoch*len(self.trainloader) + i)

                self.optimizer_weight.step()  # update weight parameters
                # unused modules back(복귀)
                self.net.unused_modules_back()  # super_proxyless.py 167번째줄

            scheduler.step()
            warmup = epoch + 1 < warmup_epochs

            (val_loss, val_top1, val_top5) = self.validate()  # validation 진행
            print(f'Loss : {losses.val:.4f}, {losses.avg:.4f}')
            print(f'Top-1 acc : {top1.val:.3f}, {top1.avg:.3f}')
            print(f'Top-5 acc : {top5.val:.3f}, {top5.avg:.3f}')
            print(f'learning rate : {scheduler.get_lr()[0]}')
            print(f'Validation Loss : {val_loss}')
            print(f'Valid Top-1 acc : {val_top1}')
            print(f'Valid Top-5 acc : {val_top5}')
            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
        return scheduler.get_lr()[0]

    def train(self, init_lr, warmup=0, train_epochs=30):
        self.optimizer_weight = optim.SGD(
            self.net.weight_params, lr=init_lr, momentum=0.9)  # optimizer 재설정
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=train_epochs+1, T_mult=1, eta_min=1e-5)  # cosine annealing
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
                self.net.MODE = "NORMAL"
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup
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
                self.net.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0 and i % 6 == 0:
                    # update architecture parameters
                    arch_loss = self.gradient_step(
                        self.net, self.optimizer_alpha)  # gradient update
                # write in tensorboard
                if i % 5 == 0:
                    # print(loss.item())
                    self.writer.add_scalar(
                        'train loss', loss.item(), epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('train_top-1 acc', top1.val,
                                           epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('train_top-5 acc', top5.val,
                                           epoch*len(self.trainloader) + i)

            scheduler.step()
            print(f'Loss : {losses.val:.4f}, {losses.avg:.4f}')
            print(f'Top-1 acc : {top1.val:.3f}, {top1.avg:.3f}')
            print(f'Top-5 acc : {top5.val:.3f}, {top5.avg:.3f}')
            print(f'learning rate : {scheduler.get_lr()[0]}')

            (val_loss, val_top1, val_top5) = self.validate()  # validation 진행
            print(f'Validation Loss : {val_loss}')
            print(f'Valid Top-1 acc : {val_top1}')
            print(f'Valid Top-5 acc : {val_top5}')

            self.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.optimizer_weight.state_dict(),
                'arch_optimizer': self.optimizer_alpha.state_dict(),
                'state_dict': self.net.state_dict(),
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
        self.net.train()
        # set chosen op active
        self.net.set_chosen_op_active()  # super_proxyless.py 175번째줄
        # remove unused modules
        self.net.unused_modules_off()

        # self.net.eval()
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

        # unused modules back
        self.net.unused_modules_back()

        return losses.avg, top1.avg, top5.avg

    def save_model(self, checkpoint):
        #checkpoint = {'state_dict': self.net.module.state_dict()}
        model_name = 'checkpoint.pth'

        # add `dataset` info to the checkpoint
        latest_fname = os.path.join("./output", 'latest.txt')
        model_path = os.path.join("./output", model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

    def gradient_step(self, net, arch_optimizer):
        #print("let update architecture parameter!")
        # switch to train mode
        net.train()
        # sample a batch of data from validation set (architecture parameter은 validation set에서 update)
        images, labels = next(iter(self.validloader))
        images, labels = images.to(self.device), labels.to(self.device)
        # compute output
        net.reset_binary_gates()  # random sample binary gates
        net.unused_modules_off()  # remove unused module for speedup
        output = net(images)
        # loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        # compute gradient and do SGD step
        # zero grads of weight_param, arch_param & binary_param
        net.zero_grad()
        loss.backward()
        # set architecture parameter gradients
        net.set_arch_param_grad()  # super_proxyless.py 137번째줄
        arch_optimizer.step()
        net.rescale_updated_arch_param()  # super_proxyless.py 145번째줄
        # back to normal mode
        net.unused_modules_back()
        #print("architecture parameter update finished")
        self.net.MODE = "NONE"
        return loss.data.item()
