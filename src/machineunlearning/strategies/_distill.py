"""SCRUB / knowledge distillation helpers.

Paper: Towards Unbounded Machine Unlearning (SCRUB) — https://arxiv.org/abs/2302.09880
Source: https://github.com/meghdadk/SCRUB
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network."""

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


def adjust_learning_rate(epoch, optimizer, lr_decay_epochs, sgda_learning_rate, lr_decay_rate):
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = sgda_learning_rate
    if steps > 0:
        new_lr = sgda_learning_rate * (lr_decay_rate**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
    return new_lr


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_distill(
    epoch,
    train_loader,
    module_list,
    swa_model,
    criterion_list,
    optimizer,
    gamma,
    alpha,
    beta,
    split,
    quiet=False,
):
    """One epoch distillation."""
    for module in module_list:
        module.train()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    for _idx, data in enumerate(train_loader):
        input, target = data
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0

        if split == "minimize":
            loss = gamma * loss_cls + alpha * loss_div + beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
        elif split == "linear" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            kd_losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if split == "minimize":
        if not quiet:
            print(f" * Acc@1 {top1.avg:.3f} ")
        return top1.avg, losses.avg
    else:
        return kd_losses.avg
