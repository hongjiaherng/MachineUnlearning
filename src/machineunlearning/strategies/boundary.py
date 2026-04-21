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


class _FGSM:
    """Single-step FGSM attack operating in the model's input space.

    We never assume a particular normalisation scheme: the attack adds
    `bound * sign(grad)` to the already-normalised input. Pixel-space
    clamping is skipped because it would silently destroy the perturbation
    on inputs normalised to ranges like [-1, 1] (e.g. MNIST).
    """

    def __init__(self, model: nn.Module, bound: float, device: torch.device):
        self.model = model
        self.bound = bound
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(device)

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad()
        x_adv = x.detach().clone().requires_grad_(True).to(self.device)
        loss = self.criterion(self.model(x_adv), y)
        loss.backward()
        return (x_adv + self.bound * x_adv.grad.data.sign()).detach()


def _inf_iter(loader):
    while True:
        yield from loader


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    p = cfg.strategy.params

    fixed = copy.deepcopy(ctx.model).to(ctx.device)
    shrunk = copy.deepcopy(ctx.model).to(ctx.device)

    attack = _FGSM(fixed, bound=p.bound, device=ctx.device)

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
