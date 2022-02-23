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


class ArchSearch:
    def __init__(self, net, data_cifar10, is_warmup, start_epoch, args):
        self.net = net
        self.data_cifar10 = data_cifar10
        self.trainloader = self.data_cifar10.train
        self.validloader = self.data_cifar10.valid
        self.testloader = self.data_cifar10.test
        self.is_warmup = is_warmup
        self.start_epoch = start_epoch
        self.valid_iter = None
        self.writer = SummaryWriter()

        self.n_warmup_epochs = args.warm_up_epoch
        self.n_epochs = args.train_epoch
        self.init_lr = args.lr
        self.resume = args.resume
        self.target_hardware = args.target_hardware
        self.reg_loss_type = args.grad_reg_loss_type
        self.ref_value = args.ref_value
        self.reg_loss_params = args.grad_reg_loss_params

        # weight parameter optimizer 정의 (SGD)
        self.optimizer_weight = optim.SGD(
            self.net.weight_parameters(), lr=self.init_lr, momentum=0.9, weight_decay=5e-4)
        # architecture parameter optimizer 정의 (Adam)
        self.optimizer_arch = optim.Adam(
            self.net.architecture_parameters(), betas=(0, 0.999), lr=0.001)
        if self.resume:
            self.load_model()
        self.multiGPU()
        print('Total FLOPs: %.1fM' % (self.net_flops() / 1e6))

    def load_model(self):
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'output'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./output/checkpoint.pth')
        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict, strict=False)

        # set new manual seed
        new_manual_seed = int(time.time())
        torch.manual_seed(new_manual_seed)
        torch.cuda.manual_seed_all(new_manual_seed)
        np.random.seed(new_manual_seed)

        self.is_warmup = checkpoint['warmup']
        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer_weight.load_state_dict(
            checkpoint['weight_optimizer'])
        self.optimizer_arch.load_state_dict(checkpoint['arch_optimizer'])

    def multiGPU(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu 사용
        self.net.to(self.device)
        self.net = nn.DataParallel(self.net)  # dataparallel
        cudnn.benchmark = True

    def warm_up(self):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=self.n_warmup_epochs + 1, T_mult=1, eta_min=0.03)  # cosine annealing
        for epoch in range(self.start_epoch, self.n_warmup_epochs):
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
            self.write_file(epoch+1, losses, top1, top5,
                            scheduler.get_last_lr()[0])  # 기록 저장
            (val_loss, val_top1, val_top5) = self.validate(epoch)  # validation 진행

            print(f'learning rate : {scheduler.get_last_lr()[0]}')

            # write architecture net
            for idx, block in enumerate(self.net.module.blocks):
                statement = f"{idx}. {block}"
                self.write_archnet(statement)

            state_dict = self.net.module.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)

            self.save_model({
                'warmup': True,
                'epoch': epoch,
                'weight_optimizer': self.optimizer_weight.state_dict(),
                'arch_optimizer': self.optimizer_arch.state_dict(),
                'state_dict': state_dict
            })

        return scheduler.get_last_lr()[0]

    def train(self):
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=self.n_epochs+1, T_mult=1, eta_min=1e-3)  # cosine annealing

        nBatch = len(self.trainloader)
        update_schedule = self.get_update_schedule(nBatch)
        for epoch in range(self.start_epoch, self.n_epochs):
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
                self.net.module.zero_grad()
                loss.backward()
                self.optimizer_weight.step()  # update weight parameters
                # unused modules back
                self.net.module.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    if i % 5 == 0:
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
            self.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.optimizer_weight.state_dict(),
                'arch_optimizer': self.optimizer_arch.state_dict(),
                'state_dict': self.net.module.state_dict(),
            })
            # self.tmp_normalnet = self.net.module.cpu().convert_to_normal_net()
            # self.write_latest_archnet(self.tmp_normalnet)
        self.save_final_model({
            'weight_optimizer': self.optimizer_weight.state_dict(),
            'arch_optimizer': self.optimizer_arch.state_dict(),
            'state_dict': self.net.module.state_dict(),
        })
        # convert to normal network according to architecture parameters
        self.normalnet = self.net.module.cpu().convert_to_normal_net()

    def validate(self, epoch):
        MixedEdge.MODE = None
        # set chosen op active
        self.net.module.set_chosen_op_active()  # super_proxyless.py 175번째줄
        # remove unused modules
        self.net.module.unused_modules_off()

        self.net.train()
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

        flops = self.net_flops()
        # unused modules back
        self.net.module.unused_modules_back()
        self.write_file_valid(losses, top1, top5, flops)

        return losses.avg, top1.avg, top5.avg

    def net_flops(self):
        data_shape = [1] + list(self.data_cifar10.data_shape)
        input_var = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, _ = self.net.module.get_flops(input_var)
        return flop

    def get_update_schedule(self, nBatch):
        schedule = {}
        for i in range(nBatch):
            if (i + 1) % 5 == 0:
                schedule[i] = 1
        return schedule

    def write_latest_archnet(self, tmp_net):
        logfile = os.path.join("./output", 'archnet.txt')
        statement = f"{tmp_net}"
        with open(logfile, 'w') as fout:
            fout.write(statement)

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

    def write_file_valid(self, losses, top1, top5, flops):
        flops = flops / 1e6
        logfile = os.path.join("./output", 'out.txt')
        tmp = f"loss : {losses.avg:.3f}, top1-accuracy : {top1.avg:.3f}, top5-accuracy : {top5.avg:.3f}, flops : {flops:.2f}M\n"
        with open(logfile, 'a') as fout:
            fout.write(tmp)

    def save_model(self, checkpoint):
        model_name = 'checkpoint.pth'
        latest_fname = os.path.join("./output", 'latest.txt')
        model_path = os.path.join("./output", model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

    def save_final_model(self, final):
        model_name = 'final.pth'
        latest_fname = os.path.join("./output", 'latest.txt')
        model_path = os.path.join("./output", model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(final, model_path)

    def gradient_step(self):
        # switch to train mode
        self.net.train()
        MixedEdge.MODE = 'two'
        # sample a batch of data from validation set (architecture parameter은 validation set에서 update)
        images, labels = self.valid_next_batch()
        images, labels = images.to(self.device), labels.to(self.device)
        # compute output
        self.net.module.reset_binary_gates()  # random sample binary gates
        self.net.module.unused_modules_off()  # remove unused module for speedup
        output = self.net(images)
        # loss
        criterion = nn.CrossEntropyLoss()
        ce_loss = criterion(output, labels)
        if self.target_hardware is None:
            expected_value = None
        elif self.target_hardware == 'flops':
            data_shape = [1] + list(self.data_cifar10.data_shape)
            input_var = torch.zeros(data_shape, device=self.device)
            expected_value = self.net.module.expected_flops(input_var)
        loss = self.add_regularization_loss(ce_loss, expected_value)
        # compute gradient and do SGD step
        # zero grads of weight_param, arch_param & binary_param
        self.net.module.zero_grad()
        loss.backward()
        # set architecture parameter gradients
        self.net.module.set_arch_param_grad()  # super_proxyless.py 137번째줄
        self.optimizer_arch.step()
        self.net.module.rescale_updated_arch_param()  # super_proxyless.py 145번째줄
        # back to normal mode
        self.net.module.unused_modules_back()
        MixedEdge.MODE = None
        return loss.data.item()

    def valid_next_batch(self):
        if self.valid_iter is None:
            self.valid_iter = iter(self.validloader)
        try:
            data = next(self.valid_iter)
        except StopIteration:
            self.valid_iter = iter(self.validloader)
            data = next(self.valid_iter)
        return data

    def add_regularization_loss(self, ce_loss, expected_value):
        if expected_value is None:
            return ce_loss

        if self.reg_loss_type == 'mul#log':
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)
            # noinspection PyUnresolvedReferences
            reg_loss = (torch.log(expected_value) /
                        math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss
        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value -
                                     self.ref_value) / self.ref_value
            return ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return ce_loss
