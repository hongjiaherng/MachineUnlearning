"""Selective Synaptic Dampening.

Paper: Fast Machine Unlearning Without Retraining Through Selective Synaptic
       Dampening — https://arxiv.org/abs/2308.07707
Source: https://github.com/if-loops/selective-synaptic-dampening

One-shot parameter surgery: compute per-parameter Fisher importance on both
the forget set and the full training distribution, then multiplicatively
dampen weights where forget-importance exceeds `α × retain-importance`.
"""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from machineunlearning.strategies.base import UnlearnContext


def _fisher_importance(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    """Mean of squared gradients per parameter — a diagonal empirical Fisher."""
    criterion = nn.CrossEntropyLoss()
    importance = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        criterion(model(x), y).backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                importance[name] += p.grad.detach().pow(2)

    n_batches = len(loader)
    return {name: imp / n_batches for name, imp in importance.items()}


def _dampen(
    model: nn.Module,
    retain_importance: dict[str, torch.Tensor],
    forget_importance: dict[str, torch.Tensor],
    selection_weighting: float,
    dampening_constant: float,
    exponent: float,
    lower_bound: float,
) -> None:
    """In-place dampening of parameters where forget mass > α × retain mass."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            r = retain_importance[name]
            f = forget_importance[name]
            selected = f > r * selection_weighting
            if not selected.any():
                continue
            weight = ((r * dampening_constant) / f).pow(exponent)
            weight = torch.where(weight > lower_bound, torch.full_like(weight, lower_bound), weight)
            p.data = torch.where(selected, p.data * weight, p.data)


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    p = cfg.strategy.params
    model = ctx.model
    model.eval()

    forget_imp = _fisher_importance(model, ctx.unlearn_loader, ctx.device)
    retain_imp = _fisher_importance(model, ctx.retain_loader, ctx.device)

    _dampen(
        model=model,
        retain_importance=retain_imp,
        forget_importance=forget_imp,
        selection_weighting=p.selection_weighting,
        dampening_constant=p.dampening_constant,
        exponent=p.exponent,
        lower_bound=p.lower_bound,
    )

    return model
