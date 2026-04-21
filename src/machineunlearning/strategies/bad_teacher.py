"""Bad Teacher / Blindspot unlearning.

Paper: Can Bad Teaching Induce Forgetting? Unlearning in Deep Networks using
       an Incompetent Teacher — https://arxiv.org/abs/2205.08096
Source: https://github.com/vikram2000b/bad-teaching-unlearning

Distils into the student a weighted mixture of two teachers: a randomly
initialised "bad" teacher on forget samples, and the fully-trained teacher on
retain samples.
"""

import copy
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from machineunlearning.data.dataset import UnlearningDataset
from machineunlearning.strategies.base import UnlearnContext


def _mixture_kl_loss(
    student_logits: torch.Tensor,
    is_forget: torch.Tensor,
    full_teacher_logits: torch.Tensor,
    bad_teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(student ‖ mixture) where the mixture is per-sample selected by `is_forget`."""
    is_forget = is_forget.unsqueeze(1)
    full = F.softmax(full_teacher_logits / temperature, dim=1)
    bad = F.softmax(bad_teacher_logits / temperature, dim=1)
    target = is_forget * bad + (1 - is_forget) * full
    log_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_student, target, reduction="batchmean")


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    params = cfg.strategy.params
    temperature = params.kl_temperature
    lr = params.lr
    batch_size = params.batch_size
    retain_frac = params.retain_frac

    student = copy.deepcopy(ctx.model)
    full_teacher = ctx.model
    bad_teacher = ctx.unlearning_teacher

    full_teacher.eval()
    bad_teacher.eval()

    retain_dataset = ctx.retain_loader.dataset
    k = int(retain_frac * len(retain_dataset))
    indices = random.sample(range(len(retain_dataset)), k)
    retain_subset = Subset(retain_dataset, indices)
    mixed = UnlearningDataset(forget_data=ctx.unlearn_loader.dataset, retain_data=retain_subset)
    loader = DataLoader(mixed, batch_size=batch_size, shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)

    for epoch in range(cfg.strategy.epochs):
        losses = []
        for x, is_forget in tqdm(loader, desc=f"Bad teacher epoch {epoch + 1}", leave=False):
            x = x.to(ctx.device)
            is_forget = is_forget.to(ctx.device).float()

            with torch.no_grad():
                full_logits = full_teacher(x)
                bad_logits = bad_teacher(x)

            student_logits = student(x)
            loss = _mixture_kl_loss(student_logits, is_forget, full_logits, bad_logits, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        tqdm.write(f"Bad teacher epoch {epoch + 1} | loss {np.mean(losses):.4f}")

    return student
