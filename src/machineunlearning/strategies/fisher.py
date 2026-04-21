"""Fisher Forgetting.

Paper: Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep
       Networks — https://arxiv.org/abs/1911.04933
Source: https://github.com/AdityaGolatkar/SelectiveForgetting

Approximates posterior covariance via the empirical Fisher (diagonal of the
Hessian) on the retain set, then replaces each parameter with a Gaussian
sample centred at the original weights with variance α / (Fisher + ε). The
row of the final classifier head corresponding to the forget class is zeroed.
"""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from machineunlearning.strategies.base import UnlearnContext


def _accumulate_fisher(model: nn.Module, dataset: Dataset, device: torch.device) -> None:
    """Attach `grad_acc` and `grad2_acc` buffers to each parameter in place."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    for p in model.parameters():
        p.grad_acc = torch.zeros_like(p.data)
        p.grad2_acc = torch.zeros_like(p.data)

    for x, y in tqdm(loader, desc="Fisher accumulation", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=-1).detach()

        for cls in range(logits.size(1)):
            target = torch.full_like(y, cls)
            model.zero_grad()
            loss_fn(logits, target).backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad_acc += (y == target).float() * p.grad.data
                    p.grad2_acc += probs[:, cls] * p.grad.data.pow(2)

    n = len(loader)
    for p in model.parameters():
        p.grad_acc /= n
        p.grad2_acc /= n


def _posterior_mean_var(
    p: nn.Parameter, forget_class: int, num_classes: int, alpha: float
) -> tuple[torch.Tensor, torch.Tensor]:
    var = (1.0 / (p.grad2_acc + 1e-8)).clamp(max=1e3)
    if p.size(0) == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var

    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()

    mu = p.data0.clone()
    if p.size(0) == num_classes:
        mu[forget_class] = 0
        var[forget_class] = 1e-4
        var *= 10  # last-layer variance boost
    elif p.ndim == 1:
        var *= 10  # BatchNorm variance boost

    return mu, var


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    alpha = cfg.strategy.params.alpha
    model = ctx.model

    for p in model.parameters():
        p.data0 = p.data.clone()

    _accumulate_fisher(model, ctx.retain_loader.dataset, ctx.device)

    for p in model.parameters():
        mu, var = _posterior_mean_var(p, ctx.unlearn_class, ctx.num_classes, alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

    return model
