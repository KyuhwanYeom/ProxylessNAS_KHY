import time
import math
import torch
import torch.nn as nn
from utils import *

class train():
    def __init__(self, net, trainloader, optimizer_weight, optimizer_alpha):
        self.net = net
        self.trainloader = trainloader
        self.optimizer_weight = optimizer_weight
        self.optimizer_alpha = optimizer_alpha

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu 사용
        # print(device)
        self.net.to(self.device)
        self.warm_up()
        self.train()

    def warm_up(self, warmup=0, warmup_epochs=25):
        lr_max = 0.05
        nBatch = len(self.trainloader)
        T_total = warmup_epochs * nBatch
        for epoch in range(warmup, warmup_epochs):
            print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
            running_loss = 0.0
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            # switch to train mode
            self.net.train()

            end = time.time()
            for i, data in enumerate(self.trainloader, 0):
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                warmup_lr = 0.5 * lr_max * \
                    (1 + math.cos(math.pi * T_cur / T_total))  # cosine annealing
                for param_group in self.optimizer_weight.param_groups:
                    param_group['lr'] = warmup_lr
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates (super_proxyless.py 131번째줄)
                # remove unused module for speedup (super_proxyless.py 153번째줄)
                self.net.unused_modules_off()
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.label_smoothing:
                    loss = cross_entropy_with_label_smoothing(output, labels, 0.1)
                else:
                    loss = nn.CrossEntropyLoss()
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                # compute gradient and do SGD step
                self.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                loss.backward()
                self.optimizer_weight.step()  # update weight parameters
                running_loss += loss.item()
                # unused modules back(복귀)
                self.net.unused_modules_back()  # super_proxyless.py 167번째줄
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            warmup = epoch + 1 < warmup_epochs
            batch_log = 'Warmup Train [{0}][{1}/{2}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                       losses=losses, top1=top1, top5=top5, lr=warmup_lr)
            print(batch_log)

            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)


    def train(self, warmup=0, warmup_epochs=25):
        nBatch = len(self.trainloader)
        # 0 ~ 120 (n_epochs = 120)
        for epoch in range(0, 120):
            print('\n', '-' * 30, 'Train epoch: %d' %
                (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.net.train()

            end = time.time()
            for i, data in enumerate(self.trainloader, 0):
                data_time.update(time.time() - end)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate(  # cosine annealing 사용
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.label_smoothing:
                    loss = cross_entropy_with_label_smoothing(output, labels, 0.1)
                else:
                    loss = nn.CrossEntropyLoss()
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss, inputs.size(0))
                top1.update(acc1[0], inputs.size(0))
                top5.update(acc5[0], inputs.size(0))
                # compute gradient and do SGD step
                # zero grads of weight_param, arch_param & binary_param
                self.net.zero_grad()
                loss.backward()
                self.optimizer.step()  # update weight parameters
                # unused modules back
                self.net.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    # update architecture parameters
                    start_time = time.time()
                    arch_loss = self.gradient_step(self.net, self.optimizer_alpha)  # gradient update
                    used_time = time.time() - start_time
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.run_manager.best_acc = max(
                    self.run_manager.best_acc, val_top1)
            # save model
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })


    def gradient_step(self, net, arch_optimizer):
        # switch to train mode
        net.train()
        # sample a batch of data from validation set (architecture parameter은 validation set에서 update)
        images, labels = self.run_manager.run_config.valid_next_batch
        images, labels = images.to(self.run_manager.device), labels.to(
            self.run_manager.device)
        # compute output
        net.reset_binary_gates()  # random sample binary gates
        net.unused_modules_off()  # remove unused module for speedup
        output = net(images)
        # loss
        loss = nn.CrossEntropyLoss()
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
        return loss.data.item()
