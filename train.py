import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

class train():
    def __init__(self, net, trainloader, validloader, testloader, optimizer_weight, optimizer_alpha):
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.optimizer_weight = optimizer_weight
        self.optimizer_alpha = optimizer_alpha
        self.lr = 0.05
        
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')  # gpu 사용
        # print(device)
        self.net.to(self.device)
        self.lr = self.warm_up()
        self.train(self.lr)
        self.test()

    def warm_up(self, warmup=0, warmup_epochs=25):
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_weight, T_max=100, eta_min=0) # cosine annealing
        nBatch = len(self.trainloader)
        lr_max = 0.05
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
                    loss = nn.CrossEntropyLoss(output, labels)
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
            #scheduler.step()
            warmup = epoch + 1 < warmup_epochs
            print(f'Loss : {losses.val:.4f}, {losses.avg:.4f}')
            print(f'Top-1 acc : {top1.val:.3f}, {top1.avg:.3f}')
            print(f'Top-5 acc : {top5.val:.3f}, {top5.avg:.3f}')
            print(f'learning rate : {warmup_lr}')
            state_dict = self.net.state_dict()
            # rm architecture parameters & binary gates
            for key in list(state_dict.keys()):
                if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                    state_dict.pop(key)
        return warmup_lr


    def train(self, init_lr, warmup=0, train_epochs=3):
        nBatch = len(self.trainloader)
        T_total = train_epochs * nBatch
        for epoch in range(0, train_epochs):
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
                print("let update weight parameter")
                data_time.update(time.time() - end)
                # lr
                T_cur = epoch * nBatch + i
                train_lr = 0.5 * init_lr * \
                    (1 + math.cos(math.pi * T_cur / T_total))  # cosine annealing
                for param_group in self.optimizer_weight.param_groups:
                    param_group['lr'] = train_lr
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # compute output
                self.net.reset_binary_gates()  # random sample binary gates
                self.net.unused_modules_off()  # remove unused module for speedup
                output = self.net(inputs)  # forward (DataParallel)
                # loss
                if self.net.label_smoothing:
                    loss = cross_entropy_with_label_smoothing(output, labels, 0.1)
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
                if epoch > 0:
                    # update architecture parameters
                    arch_loss = self.gradient_step(self.net, self.optimizer_alpha)  # gradient update
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
            print(f'Loss : {losses.val:.4f}, {losses.avg:.4f}')
            print(f'Top-1 acc : {top1.val:.3f}, {top1.avg:.3f}')
            print(f'Top-5 acc : {top5.val:.3f}, {top5.avg:.3f}')
            print(f'learning rate : {train_lr}')
            
            (val_loss, val_top1, val_top5) = self.validate()
            print(f'Validation Loss : {val_loss}')
            print(f'Valid Top-1 acc : {val_top1}')
            print(f'Valid Top-5 acc : {val_top5}')
            
    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data[0].to(self.device), data[1].to(self.device)
                # 신경망에 이미지를 통과시켜 출력을 계산합니다
                outputs = self.net(images)
                # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        
    def validate(self):
        # set chosen op active
        self.net.set_chosen_op_active()  # super_proxyless.py 175번째줄
        # remove unused modules
        self.net.unused_modules_off()
        # test on validation set under train mode
        valid_res = self.validate_validloader()
        # unused modules back
        self.net.unused_modules_back()
        return valid_res

    def validate_validloader(self):
        self.net.train()
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

    def gradient_step(self, net, arch_optimizer):
        print("let update architecture parameter!")
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
        print("architecture parameter update finished")
        return loss.data.item()
