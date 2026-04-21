"""Boundary Unlearning (Boundary Shrink).

Paper: Boundary Unlearning: Rapid Forgetting of Deep Networks via Shifting the
       Decision Boundary — https://ieeexplore.ieee.org/abstract/document/10203289
Source: https://github.com/TY-LEE-KR/Boundary-Unlearning-Code

For each forget-class sample we find the nearest-neighbour decision boundary
with a single-step FGSM attack, read off the adversarial class prediction, and
train the model to output that class instead. The boundary around the forget
class shrinks.
"""

import copy

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from machineunlearning.strategies.base import UnlearnContext

# CIFAR-10 channel stats — the FGSM perturbation operates in pixel space, so we
# need to denormalise/clip/renormalise for RGB inputs. MNIST-style 1-channel
# inputs skip this path entirely (`norm=False`).
_CIFAR_MEAN = (0.4914, 0.4822, 0.2265)
_CIFAR_STD = (0.2023, 0.1994, 0.2010)


class _FGSM:
    """Single-step FGSM attack with optional normalise/clip round-trip."""

    def __init__(self, model: nn.Module, bound: float, norm: bool, device: torch.device):
        self.model = model
        self.bound = bound
        self.norm = norm
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.norm:
            return x
        y = x.clone()
        for c, (m, s) in enumerate(zip(_CIFAR_MEAN, _CIFAR_STD)):
            y[:, c] = y[:, c] * s + m
        return y

    def _renorm(self, x: torch.Tensor) -> torch.Tensor:
        if not self.norm:
            return x
        y = x.clone()
        for c, (m, s) in enumerate(zip(_CIFAR_MEAN, _CIFAR_STD)):
            y[:, c] = (y[:, c] - m) / s
        return y

    @staticmethod
    def _discretize(x: torch.Tensor) -> torch.Tensor:
        return torch.round(x * 255) / 255

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        x_nat = self._denorm(x.detach().clone()).to(self.device)
        x_adv = x.detach().clone().requires_grad_(True).to(self.device)

        pred = self.model(x_adv)
        loss = self.criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign()
        x_adv = self._denorm(x_adv) + grad_sign * self.bound
        delta = torch.clamp(x_adv - x_nat, -self.bound, self.bound)
        x_adv = torch.clamp(x_nat + delta, 0.0, 1.0)
        return self._renorm(self._discretize(x_adv)).detach()


def _inf_iter(loader):
    while True:
        yield from loader


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    p = cfg.strategy.params

    fixed = copy.deepcopy(ctx.model).to(ctx.device)
    shrunk = copy.deepcopy(ctx.model).to(ctx.device)

    norm = ctx.num_channels == 3
    attack = _FGSM(fixed, bound=p.bound, norm=norm, device=ctx.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(shrunk.parameters(), lr=p.lr, momentum=p.momentum)

    forget_iter = _inf_iter(ctx.unlearn_loader)
    total_steps = cfg.strategy.epochs * len(ctx.unlearn_loader)

    for _ in tqdm(range(total_steps), desc="Boundary shrink"):
        x, y = next(forget_iter)
        x, y = x.to(ctx.device), y.to(ctx.device)

        fixed.eval()
        x_adv = attack.perturb(x, y)
        adv_label = torch.argmax(fixed(x_adv), dim=1)

        shrunk.train()
        optimizer.zero_grad()
        loss = criterion(shrunk(x), adv_label)
        loss.backward()
        optimizer.step()

    return shrunk
