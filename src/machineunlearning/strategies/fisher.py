"""Fisher Forgetting.

Paper: Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep
       Networks — https://arxiv.org/abs/1911.04933
Source: https://github.com/AdityaGolatkar/SelectiveForgetting

Approximates posterior covariance via the empirical Fisher (diagonal of the
Hessian) on the retain set, then replaces each parameter with a Gaussian
sample centred at the original weights with variance α / (Fisher + ε). The
row of the final classifier head corresponding to the forget class is zeroed.

Implementation note: the reference repo accumulates Fisher at batch_size=1
with a retain_graph loop over classes — tractable on small CIFAR subsets,
but hours on full MNIST. We use `torch.func.vmap(grad(...))` to fuse
per-sample gradients across a batch, giving ~100× speedup without changing
the numerical semantics. Note this implements the Fisher-noise variant of
the paper — the reference repo does not include the paper's Newton-step
scrubbing `w <- w - F^-1 ∇L_forget(w)`.
"""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from machineunlearning.strategies.base import UnlearnContext


def _accumulate_fisher(
    model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int
) -> None:
    """Attach `grad_acc` and `grad2_acc` buffers to each parameter in place.

    Semantics match the reference implementation's batch_size=1 loop: for each
    sample, accumulate the gradient at the true label (grad_acc) and the
    softmax-weighted squared per-class gradients (grad2_acc ≈ diagonal Fisher).
    """
    model.eval()
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def _loss(p_dict, x, y):
        logits = functional_call(model, (p_dict, buffers), (x.unsqueeze(0),))
        return F.cross_entropy(logits, y.unsqueeze(0))

    def _logits(p_dict, x):
        return functional_call(model, (p_dict, buffers), (x.unsqueeze(0),)).squeeze(0)

    per_sample_grad = vmap(grad(_loss), in_dims=(None, 0, 0))
    per_sample_logits = vmap(_logits, in_dims=(None, 0))

    grad_acc = {k: torch.zeros_like(v) for k, v in params.items()}
    grad2_acc = {k: torch.zeros_like(v) for k, v in params.items()}

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    n_total = 0
    num_classes: int | None = None

    for x, y in tqdm(loader, desc="Fisher accumulation", leave=False):
        x, y = x.to(device), y.to(device)
        B = x.size(0)
        n_total += B

        with torch.no_grad():
            logits = per_sample_logits(params, x)
            probs = F.softmax(logits, dim=-1)
        if num_classes is None:
            num_classes = logits.size(-1)

        for cls in range(num_classes):
            y_cls = torch.full_like(y, cls)
            g_cls = per_sample_grad(params, x, y_cls)
            mask = (y == cls).float()
            weight = probs[:, cls]
            for name in grad_acc:
                g = g_cls[name]
                shape = [B] + [1] * (g.ndim - 1)
                grad_acc[name] += (mask.view(*shape) * g).sum(dim=0)
                grad2_acc[name] += (weight.view(*shape) * g.pow(2)).sum(dim=0)

    for name, p in model.named_parameters():
        p.grad_acc = grad_acc[name] / n_total
        p.grad2_acc = grad2_acc[name] / n_total


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
    batch_size = cfg.strategy.params.get("fisher_batch_size", 128)
    model = ctx.model

    for p in model.parameters():
        p.data0 = p.data.clone()

    _accumulate_fisher(model, ctx.retain_loader.dataset, ctx.device, batch_size)

    for p in model.parameters():
        mu, var = _posterior_mean_var(p, ctx.unlearn_class, ctx.num_classes, alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()

    return model
