import time
import math
import torch
import torch.nn as nn
from utils import *

def warm_up(net, trainloader, optimizer, warmup = 0, warmup_epochs=25):
    lr_max = 0.05
    nBatch = len(trainloader)
    T_total = warmup_epochs * nBatch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # gpu 사용
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
            warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * T_cur / T_total)) # cosine annealing
            for param_group in self.run_manager.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            inputs, labels = data[0].to(device), data[1].to(device)
                # compute output
            net.reset_binary_gates()  # random sample binary gates (super_proxyless.py 131번째줄)
            net.unused_modules_off()  # remove unused module for speedup (super_proxyless.py 153번째줄)
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
            net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
            loss.backward()
            optimizer.step()  # update weight parameters
            # unused modules back(복귀)
            net.unused_modules_back() # super_proxyless.py 167번째줄
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
        
def train(net, trainloader, optimizer, warmup = 0, warmup_epochs=25):
    data_loader = self.run_manager.run_config.train_loader # self.run_manager는 앞의 arch_search_run_manager
        nBatch = len(data_loader)
        if fix_net_weights:
            data_loader = [(0, 0)] * nBatch

        arch_param_num = len(list(self.net.architecture_parameters())) # super_proxyless.py 92번째줄
        binary_gates_num = len(list(self.net.binary_gates()))
        weight_param_num = len(list(self.net.weight_parameters()))
        print(
            '#arch_params: %d\t#binary_gates: %d\t#weight_params: %d' %
            (arch_param_num, binary_gates_num, weight_param_num)
        )

        update_schedule = self.arch_search_config.get_update_schedule(nBatch)

        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.n_epochs): # 0 ~ 120 (n_epochs = 120)
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            entropy = AverageMeter()
            # switch to train mode
            self.run_manager.net.train()

            end = time.time()
            for i, (images, labels) in enumerate(data_loader):
                data_time.update(time.time() - end)
                # lr
                lr = self.run_manager.run_config.adjust_learning_rate( # cosine annealing 사용
                    self.run_manager.optimizer, epoch, batch=i, nBatch=nBatch
                )
                # network entropy
                net_entropy = self.net.entropy() # super_proxyless.py 114번째줄
                entropy.update(net_entropy.data.item() / arch_param_num, 1)
                # train weight parameters if not fix_net_weights
                if not fix_net_weights:
                    images, labels = images.to(self.run_manager.device), labels.to(self.run_manager.device)
                    # compute output
                    self.net.reset_binary_gates()  # random sample binary gates
                    self.net.unused_modules_off()  # remove unused module for speedup
                    output = self.run_manager.net(images)  # forward (DataParallel)
                    # loss
                    if self.run_manager.run_config.label_smoothing > 0:
                        loss = cross_entropy_with_label_smoothing(
                            output, labels, self.run_manager.run_config.label_smoothing
                        )
                    else:
                        loss = self.run_manager.criterion(output, labels)
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                    losses.update(loss, images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                    # compute gradient and do SGD step
                    self.run_manager.net.zero_grad()  # zero grads of weight_param, arch_param & binary_param
                    loss.backward()
                    self.run_manager.optimizer.step()  # update weight parameters
                    # unused modules back
                    self.net.unused_modules_back()
                # skip architecture parameter updates in the first epoch
                if epoch > 0:
                    # update architecture parameters according to update_schedule
                    for j in range(update_schedule.get(i, 0)):
                        start_time = time.time()
                        if isinstance(self.arch_search_config, RLArchSearchConfig):
                            reward_list, net_info_list = self.rl_update_step(fast=True)
                            used_time = time.time() - start_time
                            log_str = 'REINFORCE [%d-%d]\tTime %.4f\tMean Reward %.4f\t%s' % (
                                epoch + 1, i, used_time, sum(reward_list) / len(reward_list), net_info_list
                            )
                            self.write_log(log_str, prefix='rl', should_print=False)
                        elif isinstance(self.arch_search_config, GradientArchSearchConfig):
                            arch_loss, exp_value = self.gradient_step() # gradient update
                            used_time = time.time() - start_time
                            log_str = 'Architecture [%d-%d]\t Time %.4f\t Loss %.4f\t %s %s' % \
                                      (epoch + 1, i, used_time, arch_loss,
                                       self.arch_search_config.target_hardware, exp_value)
                            self.write_log(log_str, prefix='gradient', should_print=False)
                        else:
                            raise ValueError('do not support: %s' % type(self.arch_search_config))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # training log
                if i % self.run_manager.run_config.print_frequency == 0 or i + 1 == nBatch:
                    batch_log = 'Train [{0}][{1}/{2}]\t' \
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                                'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                                'Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t' \
                                'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})\t' \
                                'Top-5 acc {top5.val:.3f} ({top5.avg:.3f})\tlr {lr:.5f}'. \
                        format(epoch + 1, i, nBatch - 1, batch_time=batch_time, data_time=data_time,
                               losses=losses, entropy=entropy, top1=top1, top5=top5, lr=lr)
                    self.run_manager.write_log(batch_log, 'train')

            # print current network architecture
            self.write_log('-' * 30 + 'Current Architecture [%d]' % (epoch + 1) + '-' * 30, prefix='arch')
            for idx, block in enumerate(self.net.blocks):
                self.write_log('%d. %s' % (idx, block.module_str), prefix='arch')
            self.write_log('-' * 60, prefix='arch')

            # validate
            if (epoch + 1) % self.run_manager.run_config.validation_frequency == 0:
                (val_loss, val_top1, val_top5), flops, latency = self.validate()
                self.run_manager.best_acc = max(self.run_manager.best_acc, val_top1)
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})\ttop-5 acc {5:.3f}\t' \
                          'Train top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t' \
                          'Entropy {entropy.val:.5f}\t' \
                          'Latency-{6}: {7:.3f}ms\tFlops: {8:.2f}M'. \
                    format(epoch + 1, self.run_manager.run_config.n_epochs, val_loss, val_top1,
                           self.run_manager.best_acc, val_top5, self.arch_search_config.target_hardware,
                           latency, flops / 1e6, entropy=entropy, top1=top1, top5=top5)
                self.run_manager.write_log(val_log, 'valid')
            # save model
            self.run_manager.save_model({
                'warmup': False,
                'epoch': epoch,
                'weight_optimizer': self.run_manager.optimizer.state_dict(),
                'arch_optimizer': self.arch_optimizer.state_dict(),
                'state_dict': self.net.state_dict()
            })

        # convert to normal network according to architecture parameters
        normal_net = self.net.cpu().convert_to_normal_net()
        print('Total training params: %.2fM' % (count_parameters(normal_net) / 1e6))
        os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)
        json.dump(normal_net.config, open(os.path.join(self.run_manager.path, 'learned_net/net.config'), 'w'), indent=4)
        json.dump(
            self.run_manager.run_config.config,
            open(os.path.join(self.run_manager.path, 'learned_net/run.config'), 'w'), indent=4,
        )
        torch.save(
            {'state_dict': normal_net.state_dict(), 'dataset': self.run_manager.run_config.dataset},
            os.path.join(self.run_manager.path, 'learned_net/init')
        )