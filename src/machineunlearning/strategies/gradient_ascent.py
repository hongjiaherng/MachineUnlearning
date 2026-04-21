"""Gradient ascent on the forget set.

Paper: Unrolling SGD: Understanding Factors Influencing Machine Unlearning
       https://arxiv.org/abs/2109.13398
Source: https://github.com/cleverhans-lab/unrolling-sgd

Maximises cross-entropy on the forget samples (via a `1 - loss` surrogate) so
the model drifts away from correctly classifying them.
"""

import copy

import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

from machineunlearning.strategies.base import UnlearnContext


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    model = copy.deepcopy(ctx.model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.5)
    loss_func = nn.CrossEntropyLoss().to(ctx.device)

    for _ in tqdm(range(cfg.strategy.epochs), desc="Gradient ascent"):
        model.train()
        for images, labels in ctx.unlearn_loader:
            images = images.to(ctx.device)
            labels = labels.long().to(ctx.device)
            optimizer.zero_grad()
            output = model(images)
            loss = 1 - loss_func(output, labels)
            loss.backward()
            optimizer.step()

    return model
