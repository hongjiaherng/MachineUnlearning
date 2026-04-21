"""Shared types for unlearning strategies.

Every strategy implements `unlearn(cfg, ctx) -> nn.Module`, where `cfg` is the
Hydra run config and `ctx` bundles the artefacts the strategy operates on.
Strategy-specific hyperparameters live under `cfg.strategy.params` and are
read by the individual strategy modules.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class UnlearnContext:
    model: nn.Module
    unlearning_teacher: nn.Module
    unlearn_class: int
    unlearn_loader: DataLoader
    retain_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    num_channels: int
    device: torch.device
