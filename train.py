import time
import math
import torch
import torch.nn as nn
from utils import *


def warm_up(net, trainloader, optimizer, warmup=0, warmup_epochs=25):
    lr_max = 0.05
    nBatch = len(trainloader)
    T_total = warmup_epochs * nBatch
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu 사용
    # print(device)
    net.to(device)
    for epoch in range(warmup, warmup_epochs):
        print('\n', '-' * 30, 'Warmup epoch: %d' % (epoch + 1), '-' * 30, '\n')
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        net.train()

        end = time.time()
        for i, data in enumerate(trainloader, 0):
            data_time.update(time.time() - end)
            # lr
            T_cur = epoch * nBatch + i
            warmup_lr = 0.5 * lr_max * \
                (1 + math.cos(math.pi * T_cur / T_total))  # cosine annealing
            inputs, labels = data[0].to(device), data[1].to(device)
            # compute output
            net.reset_binary_gates()  # random sample binary gates (super_proxyless.py 131번째줄)
            net.unused_modules_off()  # remove unused module for speedup (super_proxyless.py 153번째줄)
            output = net(inputs)  # forward (DataParallel)
            # loss
            if net.label_smoothing:
                loss = cross_entropy_with_label_smoothing(output, labels, 0.1)
            else:
                loss = nn.CrossEntropyLoss()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            # compute gradient and do SGD step
            net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
            loss.backward()
            optimizer.step()  # update weight parameters
            # unused modules back(복귀)
            net.unused_modules_back()  # super_proxyless.py 167번째줄
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        warmup = epoch + 1 < warmup_epochs

        state_dict = net.state_dict()
        # rm architecture parameters & binary gates
        for key in list(state_dict.keys()):
            if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                state_dict.pop(key)
        checkpoint = {
            'state_dict': state_dict,
            'warmup': warmup,
        }
        if warmup:
            checkpoint['warmup_epoch'] = epoch,
        self.run_manager.save_model(checkpoint, model_name='warmup.pth.tar')


def train(net, trainloader, optimizer, warmup=0, warmup_epochs=25):
    nBatch = len(trainloader)
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu 사용
    # print(device)
    net.to(device)
    arch_param_num = len(list(net.architecture_parameters()))
    binary_gates_num = len(list(net.binary_gates()))
    weight_param_num = len(list(net.weight_parameters()))
    update_schedule = arch_search_config.get_update_schedule(nBatch)
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
        net.train()

        end = time.time()
        for i, data in enumerate(trainloader, 0):
            data_time.update(time.time() - end)
            # lr
            lr = self.run_manager.run_config.adjust_learning_rate(  # cosine annealing 사용
                self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
            )
            inputs, labels = data[0].to(device), data[1].to(device)
            # compute output
            net.reset_binary_gates()  # random sample binary gates
            net.unused_modules_off()  # remove unused module for speedup
            output = net(inputs)  # forward (DataParallel)
            # loss
            if self.run_manager.run_config.label_smoothing > 0:
                loss = cross_entropy_with_label_smoothing(
                    output, labels, self.run_manager.run_config.label_smoothing
                )
            else:
                loss = nn.CrossEntropyLoss()
                # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss, inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            # compute gradient and do SGD step
            # zero grads of weight_param, arch_param & binary_param
            net.zero_grad()
            loss.backward()
            optimizer.step()  # update weight parameters
            # unused modules back
            net.unused_modules_back()
            # skip architecture parameter updates in the first epoch
            if epoch > 0:
                # update architecture parameters according to update_schedule
                for j in range(update_schedule.get(i, 0)):
                    start_time = time.time()
                    arch_loss = gradient_step(net)  # gradient update
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


def gradient_step(net, arch_optimizer):
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
