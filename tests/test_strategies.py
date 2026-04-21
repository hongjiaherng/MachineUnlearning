"""End-to-end tests for every unlearning strategy.

Each test:
  1. Builds a tiny UnlearnContext (see conftest.py).
  2. Constructs a minimal DictConfig for the strategy.
  3. Dispatches via STRATEGY_REGISTRY and asserts the result is a trained nn.Module
     with a sane output shape.

These are correctness/smoke tests — they don't assert unlearning quality.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from machineunlearning.strategies import STRATEGY_REGISTRY


def _strategy_cfg(name: str, epochs: int = 1, params: dict | None = None):
    cfg = {"strategy": {"name": name, "epochs": epochs, "params": params or {}}}
    return OmegaConf.create(cfg)


STRATEGY_PARAMS: dict[str, dict] = {
    "baseline": {"epochs": 0, "params": {}},
    "retrain": {"epochs": 1, "params": {}},
    "fine_tune": {"epochs": 1, "params": {}},
    "gradient_ascent": {"epochs": 1, "params": {}},
    "amnesiac": {"epochs": 1, "params": {}},
    "bad_teacher": {
        "epochs": 1,
        "params": {"kl_temperature": 1.0, "lr": 1e-3, "batch_size": 8, "retain_frac": 0.5},
    },
    "scrub": {
        "epochs": 1,
        "params": {
            "gamma": 0.99, "alpha": 0.001, "beta": 0.0, "msteps": 1, "kd_t": 4,
            "sgda_learning_rate": 5e-4, "lr_decay_epochs": [3, 5, 9],
            "lr_decay_rate": 0.1, "sgda_weight_decay": 5e-4, "sgda_momentum": 0.9,
        },
    },
    "boundary": {"epochs": 1, "params": {"bound": 0.1, "lr": 1e-3, "momentum": 0.9}},
    "fisher": {"epochs": 0, "params": {"alpha": 1e-6}},
    "ssd": {
        "epochs": 0,
        "params": {
            "lower_bound": 1, "exponent": 1, "dampening_constant": 1, "selection_weighting": 10,
        },
    },
    "unsir": {
        "epochs": 1,
        "params": {"noise_batch_size": 4, "num_samples": 4, "noise_train_epochs": 2},
    },
    "ntk": {"epochs": 0, "params": {"weight_decay": 0.1}},
}


@pytest.mark.parametrize("name", list(STRATEGY_PARAMS.keys()))
def test_strategy_runs(name: str, unlearn_context):
    cfg = _strategy_cfg(name, **STRATEGY_PARAMS[name])

    unlearned = STRATEGY_REGISTRY[name](cfg, unlearn_context)

    assert isinstance(unlearned, nn.Module)

    x, _ = next(iter(unlearn_context.test_loader))
    x = x.to(unlearn_context.device)
    unlearned.eval()
    with torch.no_grad():
        out = unlearned(x)
    assert out.shape == (x.size(0), unlearn_context.num_classes)
    assert torch.isfinite(out).all()


def test_registry_covers_all_strategies():
    assert set(STRATEGY_REGISTRY) == set(STRATEGY_PARAMS)
