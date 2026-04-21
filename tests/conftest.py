"""Shared pytest fixtures: tiny synthetic image dataset + pre-built UnlearnContext.

Kept intentionally tiny so the whole suite runs in seconds even on CPU. The
point of these tests is to exercise code paths end-to-end (shape bugs, import
bugs, API drift), not to measure unlearning quality.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from machineunlearning.strategies.base import UnlearnContext

NUM_CLASSES = 3
NUM_CHANNELS = 1
IMG_SIZE = 8
SAMPLES_PER_CLASS = 16
BATCH_SIZE = 8
FORGET_CLASS = 0


class _TinyCNN(nn.Module):
    """Minimal CNN that accepts (N, C, 8, 8) and returns (N, NUM_CLASSES)."""

    def __init__(self, num_classes: int = NUM_CLASSES, input_channels: int = NUM_CHANNELS):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class _TinyImageDataset(Dataset):
    """Matches torchvision convention: returns (Tensor, int)."""

    def __init__(self, images: torch.Tensor, labels: list[int]):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def _make_synthetic_dataset(seed: int = 0) -> _TinyImageDataset:
    """Class-conditional Gaussian blobs on 8x8 single-channel images."""
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for cls in range(NUM_CLASSES):
        mean = (cls - 1) * 0.5
        x = torch.randn(SAMPLES_PER_CLASS, NUM_CHANNELS, IMG_SIZE, IMG_SIZE, generator=g) + mean
        xs.append(x)
        ys.extend([cls] * SAMPLES_PER_CLASS)
    return _TinyImageDataset(torch.cat(xs), ys)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_dataset() -> _TinyImageDataset:
    return _make_synthetic_dataset()


@pytest.fixture
def trained_model(tiny_dataset: _TinyImageDataset, device: torch.device) -> nn.Module:
    """A tiny model trained for a few steps so it has meaningful weights."""
    torch.manual_seed(0)
    model = _TinyCNN().to(device)
    loader = DataLoader(tiny_dataset, batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(3):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    return model


@pytest.fixture
def unlearn_context(trained_model: nn.Module, tiny_dataset: _TinyImageDataset, device: torch.device) -> UnlearnContext:
    """Build UnlearnContext with retain/forget splits around FORGET_CLASS."""
    labels = tiny_dataset.labels
    retain_idx = [i for i, y in enumerate(labels) if y != FORGET_CLASS]
    forget_idx = [i for i, y in enumerate(labels) if y == FORGET_CLASS]

    retain = torch.utils.data.Subset(tiny_dataset, retain_idx)
    forget = torch.utils.data.Subset(tiny_dataset, forget_idx)

    retain_loader = DataLoader(retain, batch_size=BATCH_SIZE, shuffle=True)
    unlearn_loader = DataLoader(forget, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(tiny_dataset, batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(1)
    teacher = _TinyCNN().to(device)

    return UnlearnContext(
        model=copy.deepcopy(trained_model),
        unlearning_teacher=teacher,
        unlearn_class=FORGET_CLASS,
        unlearn_loader=unlearn_loader,
        retain_loader=retain_loader,
        test_loader=test_loader,
        num_classes=NUM_CLASSES,
        num_channels=NUM_CHANNELS,
        device=device,
    )
