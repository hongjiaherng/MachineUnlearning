"""Baseline: return the victim model unchanged.

Used as a no-op oracle — metrics on this should match the original trained
model and show no forgetting.
"""

from omegaconf import DictConfig
from torch import nn

from machineunlearning.strategies.base import UnlearnContext


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    del cfg
    return ctx.model
