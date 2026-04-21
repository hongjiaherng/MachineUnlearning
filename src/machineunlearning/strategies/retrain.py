"""Retrain from scratch on the retain set — gold-standard unlearning oracle."""

from omegaconf import DictConfig
from torch import nn

from machineunlearning.strategies._training import training_optimization
from machineunlearning.strategies.base import UnlearnContext


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    return training_optimization(
        model=ctx.model,
        train_loader=ctx.retain_loader,
        test_loader=ctx.test_loader,
        epochs=cfg.strategy.epochs,
        device=ctx.device,
        desc="Retraining from scratch",
    )
