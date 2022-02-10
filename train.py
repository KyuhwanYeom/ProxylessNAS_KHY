import time
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from utils import *
from mixed_op import *


class train():
    def __init__(self, net, trainloader, validloader, testloader, optimizer_weight, optimizer_alpha, is_warmup, best_acc, start_epoch):
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.optimizer_weight = optimizer_weight
        self.optimizer_alpha = optimizer_alpha
        self.is_warmup = is_warmup
        self.best_acc = best_acc
        self.start_epoch = start_epoch
        self.lr = optimizer_weight.param_groups[0]['lr']
        self.train_MODE = 'False'
        self.writer = SummaryWriter()

        self.multiGPU()

        if is_warmup == True:
            self.lr = self.warm_up()
        self.train(self.lr)

    def multiGPU(self):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # gpu 사용
        self.net.to(self.device)
        self.net = nn.DataParallel(self.net)  # dataparallel
        cudnn.benchmark = True

    def warm_up(self, warmup_epochs=1):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=warmup_epochs + 1, T_mult=1, eta_min=0.03)  # cosine annealing
        for epoch in range(self.start_epoch, warmup_epochs):
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
                self.net.module.reset_binary_gates()  # random sample binary gates
                # remove unused module for speedup
                self.net.module.unused_modules_off()
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.module.label_smoothing:
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

                self.optimizer_weight.step()  # update weight parameters
                # unused modules back(복귀)
                self.net.module.unused_modules_back()  # super_proxyless.py 167번째줄
                progress_bar(i, len(self.trainloader), 'Train Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                             % (losses.avg, top1.avg, top5.avg))

            scheduler.step()
            warmup = epoch + 1 < warmup_epochs

            self.write_file(epoch+1, losses, top1, top5,
                            scheduler.get_last_lr()[0])  # 기록 저장
            (val_loss, val_top1, val_top5) = self.validate(epoch)  # validation 진행

            print(f'learning rate : {scheduler.get_last_lr()[0]}')

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)

            if val_top1 > self.best_acc:
                self.save_model({
                    'warmup': True,
                    'epoch': epoch,
                    'acc': val_top1,
                    'weight_optimizer': self.optimizer_weight.state_dict(),
                    'arch_optimizer': self.optimizer_alpha.state_dict(),
                    'net': self.net.state_dict(),
                })
                self.best_acc = val_top1

        return scheduler.get_last_lr()[0]

    def train(self, init_lr, train_epochs=1):
        self.optimizer_weight = optim.SGD(
            self.net.module.weight_params, lr=init_lr, momentum=0.9)  # optimizer 재설정
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=train_epochs+1, T_mult=1, eta_min=1e-2)  # cosine annealing
        for epoch in range(self.start_epoch, train_epochs):
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
                self.net.MODE = 'NORMAL'
                self.net.module.reset_binary_gates()  # random sample binary gates
                self.net.module.unused_modules_off()  # remove unused module for speedup
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.module.label_smoothing:
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
                self.net.module.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0 and i % 6 == 0:
                    # update architecture parameters
                    arch_loss = self.gradient_step()  # gradient update
                # write in tensorboard
                if i % 5 == 0:
                    # print(loss.item())
                    self.writer.add_scalar(
                        'train loss', loss.item(), epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('train_top-1 acc', top1.val,
                                           epoch*len(self.trainloader) + i)

                progress_bar(i, len(self.trainloader), 'Train Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                             % (losses.avg, top1.avg, top5.avg))
            self.train_MODE = 'True'
            scheduler.step()
            self.write_file(epoch + 1, losses, top1, top5,
                            scheduler.get_last_lr()[0])
            print(f'learning rate : {scheduler.get_last_lr()[0]}')

            print_epoch = f"----------------------epoch : {epoch + 1}----------------------\n"
            # self.write_archnet(print_epoch)
            # for idx, block in enumerate(self.net.blocks):
            #    statement = f"{idx}. {block.module_str}"
            #    self.write_archnet(statement)

            (val_loss, val_top1, val_top5) = self.validate(epoch)  # validation 진행
            self.train_MODE = 'False'
            if val_top1 > self.best_acc:
                self.save_model({
                    'warmup': False,
                    'epoch': epoch,
                    'acc': val_top1,
                    'weight_optimizer': self.optimizer_weight.state_dict(),
                    'arch_optimizer': self.optimizer_alpha.state_dict(),
                    'net': self.net.state_dict(),
                })
                self.best_acc = val_top1
        # convert to normal network according to architecture parameters
        self.normalnet = self.net.module.cpu().convert_to_normal_net()
        self.save_final_model({
            'weight_optimizer': self.optimizer_weight.state_dict(),
            'arch_optimizer': self.optimizer_alpha.state_dict(),
            'net': self.normalnet.state_dict(),
        })
        print(self.normalnet)

    def validate(self, epoch):
        if self.train_MODE == 'True':
            MixedEdge.MODE = 'None'
        # set chosen op active
        self.net.module.set_chosen_op_active()  # super_proxyless.py 175번째줄
        # remove unused modules
        self.net.module.unused_modules_off()

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
                # write in tensorboard
                if i % 5 == 0:
                    # print(loss.item())
                    self.writer.add_scalar(
                        'val loss', loss.item(), epoch*len(self.validloader) + i)
                    self.writer.add_scalar('val_top-1 acc', top1.val,
                                           epoch*len(self.validloader) + i)

                progress_bar(i, len(self.validloader), 'Valid Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                             % (losses.avg, top1.avg, top5.avg))

        # unused modules back
        self.net.module.unused_modules_back()
        self.write_file_valid(losses, top1, top5)

        return losses.avg, top1.avg, top5.avg

    def write_archnet(self, statement):
        logfile = os.path.join("./output", 'archnet.txt')
        with open(logfile, 'a') as fout:
            fout.write(statement)

    def write_file(self, epoch, losses, top1, top5, lr):
        logfile = os.path.join("./output", 'out.txt')
        print_epoch = f"----------------------epoch : {epoch}----------------------\n"
        tmp = f"loss : {losses.avg:.3f}, top1-accuracy : {top1.avg:.3f}, top5-accuracy : {top5.avg:.3f}, learning rate : {lr}\n"
        with open(logfile, 'a') as fout:
            fout.write(print_epoch)
            fout.write(tmp)

    def write_file_valid(self, losses, top1, top5):
        logfile = os.path.join("./output", 'out.txt')
        tmp = f"loss : {losses.avg:.3f}, top1-accuracy : {top1.avg:.3f}, top5-accuracy : {top5.avg:.3f}\n"
        with open(logfile, 'a') as fout:
            fout.write(tmp)

    def save_model(self, checkpoint):
        model_name = 'checkpoint.pth'

        # add `dataset` info to the checkpoint
        latest_fname = os.path.join("./output", 'latest.txt')
        model_path = os.path.join("./output", model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

    def save_final_model(self, final):
        model_name = 'final.pth'

        # add `dataset` info to the checkpoint
        latest_fname = os.path.join("./output", 'latest.txt')
        model_path = os.path.join("./output", model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(final, model_path)

    def gradient_step(self):
        # switch to train mode
        self.net.train()
        # sample a batch of data from validation set (architecture parameter은 validation set에서 update)
        images, labels = next(iter(self.trainloader))
        images, labels = images.to(self.device), labels.to(self.device)
        # compute output
        self.net.module.reset_binary_gates()  # random sample binary gates
        self.net.module.unused_modules_off()  # remove unused module for speedup
        output = self.net(images)
        # loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        # compute gradient and do SGD step
        # zero grads of weight_param, arch_param & binary_param
        self.net.zero_grad()
        loss.backward()
        # set architecture parameter gradients
        self.net.module.set_arch_param_grad()  # super_proxyless.py 137번째줄
        self.optimizer_alpha.step()
        self.net.module.rescale_updated_arch_param()  # super_proxyless.py 145번째줄
        # back to normal mode
        self.net.module.unused_modules_back()
        return loss.data.item()
