"""SCRUB — Towards Unbounded Machine Unlearning.

Paper: https://arxiv.org/abs/2302.09880
Source: https://github.com/meghdadk/SCRUB

Alternates two distillation phases against a frozen teacher:
  * **Maximise** KL between student and teacher on the forget set (push away).
  * **Minimise** KL + cross-entropy on the retain set (stay aligned elsewhere).
The maximise phase runs for the first `msteps` epochs, then minimise-only.
"""

import copy
from typing import Literal

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from machineunlearning.strategies.base import UnlearnContext


class DistillKL(nn.Module):
    """Temperature-scaled KL distillation loss."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        log_p_s = F.log_softmax(student_logits / self.temperature, dim=1)
        p_t = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(log_p_s, p_t, reduction="sum") * (self.temperature**2) / student_logits.size(0)


def _decay_lr(epoch: int, optimizer: torch.optim.Optimizer, decay_epochs, base_lr: float, decay_rate: float) -> None:
    steps = sum(1 for e in decay_epochs if epoch > e)
    new_lr = base_lr * (decay_rate**steps) if steps else base_lr
    for group in optimizer.param_groups:
        group["lr"] = new_lr


def _run_distill_epoch(
    loader,
    student: nn.Module,
    teacher: nn.Module,
    criterion_cls: nn.Module,
    criterion_div: nn.Module,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    alpha: float,
    device: torch.device,
    mode: Literal["minimize", "maximize"],
) -> None:
    student.train()
    teacher.eval()

    for images, labels in loader:
        images = images.to(device).float()
        labels = labels.to(device)

        logit_s = student(images)
        with torch.no_grad():
            logit_t = teacher(images)

        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)

        loss = gamma * loss_cls + alpha * loss_div if mode == "minimize" else -loss_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    p = cfg.strategy.params

    student = copy.deepcopy(ctx.model).to(ctx.device)
    teacher = copy.deepcopy(ctx.unlearning_teacher).to(ctx.device)

    criterion_cls = nn.CrossEntropyLoss().to(ctx.device)
    criterion_div = DistillKL(p.kd_t).to(ctx.device)

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=p.sgda_learning_rate,
        momentum=p.sgda_momentum,
        weight_decay=p.sgda_weight_decay,
    )

    for epoch in tqdm(range(1, cfg.strategy.epochs + 1), desc="SCRUB"):
        _decay_lr(epoch, optimizer, p.lr_decay_epochs, p.sgda_learning_rate, p.lr_decay_rate)

        if epoch <= p.msteps:
            _run_distill_epoch(
                ctx.unlearn_loader, student, teacher,
                criterion_cls, criterion_div, optimizer,
                p.gamma, p.alpha, ctx.device, mode="maximize",
            )
        _run_distill_epoch(
            ctx.retain_loader, student, teacher,
            criterion_cls, criterion_div, optimizer,
            p.gamma, p.alpha, ctx.device, mode="minimize",
        )

    return student
