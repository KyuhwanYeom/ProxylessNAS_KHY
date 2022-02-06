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
    def __init__(self, net, trainloader, validloader, testloader, optimizer_weight, best_acc, start_epoch):
        self.net = net
        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.optimizer_weight = optimizer_weight
        self.best_acc = best_acc
        self.start_epoch = start_epoch
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

    def train(self, train_epochs=200):
        nBatch = len(self.trainloader)
        self.optimizer_weight.param_groups[0]['lr'] = 0.01
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer_weight, T_0=train_epochs + 1, T_mult=1, eta_min=1e-3)  # cosine annealing
        for epoch in range(self.start_epoch, train_epochs):
            print('\n', '-' * 30, 'epoch: %d' %
                  (epoch + 1), '-' * 30, '\n')
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            # switch to train mode
            self.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(self.trainloader):
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
                # write in tensorboard
                if i % 5 == 0:
                    # print(loss.item())
                    self.writer.add_scalar(
                        'training_train loss', loss.item(), epoch*len(self.trainloader) + i)
                    self.writer.add_scalar('training_train_top-1 acc', top1.val,
                                            epoch*len(self.trainloader) + i)
                    
                progress_bar(i, len(self.trainloader), 'Train Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                                % (losses.avg, top1.avg, top5.avg))         
            scheduler.step()
            self.write_file(epoch+1, losses, top1, top5, scheduler.get_last_lr()[0]) # 기록 저장   
            (val_loss, val_top1, val_top5) = self.validate(epoch)  # validation 진행
            
            print(f'learning rate : {scheduler.get_last_lr()[0]}')
            
            if val_top1 > self.best_acc:
                self.save_model({
                    'warmup': True,
                    'epoch': epoch,
                    'acc': val_top1,
                    'weight_optimizer': self.optimizer_weight.state_dict(),
                    'net': self.net.state_dict(),
                })
                self.best_acc = val_top1

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

    def validate(self, epoch):
        # test on validation set under train mode
        valid_res = self.validate_validloader(epoch)
        return valid_res

    def validate_validloader(self, epoch):
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
                        'training_valid loss', loss.item(), epoch*len(self.validloader) + i)
                    self.writer.add_scalar('training_valid_top-1 acc', top1.val,
                                            epoch*len(self.validloader) + i)
                progress_bar(i, len(self.validloader), 'Valid Loss: %.3f | Top-1 Acc: %.3f%% | Top-5 Acc: %.3f%%'
                             % (losses.avg, top1.avg, top5.avg))
                
        self.write_file_valid(losses, top1, top5)
        return losses.avg, top1.avg, top5.avg

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
        #checkpoint = {'state_dict': self.net.module.state_dict()}
        model_name = 'checkpoint.pth'

        # add `dataset` info to the checkpoint
        latest_fname = os.path.join("./output", 'latest.txt')
        model_path = os.path.join("./output", model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)