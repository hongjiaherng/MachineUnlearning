"""Amnesiac Unlearning.

Paper: Amnesiac Machine Learning — https://arxiv.org/abs/2010.10981
Source: https://github.com/lmgraves/AmnesiacML

Relabels every forget-class sample to a random non-forget class, concatenates
with the retain set, and fine-tunes. The model learns to predict the wrong
(but plausible) class on forget samples.
"""

import random

from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from machineunlearning.strategies._training import training_optimization
from machineunlearning.strategies.base import UnlearnContext


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    non_forget_labels = [c for c in range(ctx.num_classes) if c != ctx.unlearn_class]

    relabeled = [(x, random.choice(non_forget_labels)) for x, _ in ctx.unlearn_loader.dataset]
    retained = [(x, y) for x, y in ctx.retain_loader.dataset]
    mixed_loader = DataLoader(relabeled + retained, batch_size=128, shuffle=True, pin_memory=True)

    return training_optimization(
        model=ctx.model,
        train_loader=mixed_loader,
        test_loader=ctx.test_loader,
        epochs=cfg.strategy.epochs,
        device=ctx.device,
        desc="Amnesiac unlearning",
    )
