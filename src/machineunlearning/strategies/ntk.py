"""NTK-based scrubbing (Forgetting Outside the Box).

Paper: https://arxiv.org/abs/2003.02960
Source: https://github.com/AdityaGolatkar/SelectiveForgetting

Closed-form weight update derived from the empirical Neural Tangent Kernel
over the retain and forget sets. Expensive: builds a per-sample Jacobian with
batch_size=1 passes.
"""

from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from machineunlearning.strategies.base import UnlearnContext


def _subsample(dataset, k: int | None):
    if k is None or k >= len(dataset):
        return dataset
    idx = torch.randperm(len(dataset))[:k].tolist()
    return Subset(dataset, idx)


def _sample_jacobian_and_residual(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    subsample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (J^T, f0 - y) accumulated sample-wise at batch_size 1.

    Stored in float32 — float64 doubles memory with no material benefit here.
    """
    model.eval()
    dataset = _subsample(loader.dataset, subsample)
    single_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = next(model.parameters()).device

    jacobian_rows: list[np.ndarray] = []
    residuals: list[np.ndarray] = []

    for x, y in tqdm(single_loader, leave=False):
        x, y = x.to(device), y.cpu().numpy()
        output = model(x)
        for cls in range(num_classes):
            grads = torch.autograd.grad(output[0, cls], model.parameters(), retain_graph=True)
            jacobian_rows.append(
                np.concatenate([g.view(-1).cpu().numpy().astype(np.float32) for g in grads])
            )
        p = F.softmax(output, dim=1).detach().cpu().numpy().astype(np.float32).T
        p[y] -= 1
        residuals.append(deepcopy(p))

    return np.stack(jacobian_rows).T, np.vstack(residuals)


def _solve_closed_form(G: np.ndarray, residual: np.ndarray, n_samples: int, weight_decay: float) -> np.ndarray:
    """Solve w = -G (Gᵀ G + n·λ·I)⁻¹ residual."""
    theta = G.T @ G + n_samples * weight_decay * np.eye(G.shape[1], dtype=G.dtype)
    return -G @ np.linalg.solve(theta, residual)


def _delta_to_state_dict(delta_w: np.ndarray, model: nn.Module) -> OrderedDict:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    cursor = 0
    for name, p in model.named_parameters():
        n_params = int(np.prod(p.shape))
        chunk = delta_w[cursor : cursor + n_params]
        result[name] = torch.tensor(chunk, dtype=torch.float32).view_as(p)
        cursor += n_params
    return result


def _flatten_params(model: nn.Module) -> np.ndarray:
    return np.concatenate([p.data.view(-1).cpu().numpy() for p in model.parameters()])


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    weight_decay = cfg.strategy.params.weight_decay
    retain_subsample = cfg.strategy.params.get("retain_subsample", None)
    forget_subsample = cfg.strategy.params.get("forget_subsample", None)

    model_init = deepcopy(ctx.model)
    G_r, res_r = _sample_jacobian_and_residual(
        deepcopy(ctx.model), ctx.retain_loader, ctx.num_classes, retain_subsample
    )
    G_f, res_f = _sample_jacobian_and_residual(
        deepcopy(ctx.model), ctx.unlearn_loader, ctx.num_classes, forget_subsample
    )

    G = np.concatenate([G_r, G_f], axis=1)
    res = np.concatenate([res_r, res_f])
    n_retain = G_r.shape[1] // ctx.num_classes
    n_forget = G_f.shape[1] // ctx.num_classes
    n_total = n_retain + n_forget

    w_complete = _solve_closed_form(G, res, n_total, weight_decay).squeeze()
    w_retain = _solve_closed_form(G_r, res_r, n_retain, weight_decay).squeeze()

    delta_w = w_retain - w_complete
    pred_error = _flatten_params(ctx.model) - _flatten_params(model_init) - w_retain

    inner = np.inner(delta_w / np.linalg.norm(delta_w), pred_error / np.linalg.norm(pred_error))
    angle = np.arccos(inner) - np.pi / 2 if inner < 0 else np.arccos(inner)
    trig = np.sin(angle) if inner < 0 else np.cos(angle)
    predicted_norm = np.linalg.norm(delta_w) + 2 * trig * np.linalg.norm(pred_error)
    scale = predicted_norm / np.linalg.norm(delta_w)

    update = _delta_to_state_dict(delta_w, ctx.model)
    for name, p in ctx.model.named_parameters():
        p.data += (update[name] * scale).to(ctx.device)

    return ctx.model
