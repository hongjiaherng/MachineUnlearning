"""UNSIR — Selective Impair and Repair.

Paper: Fast Yet Effective Machine Unlearning — https://arxiv.org/abs/2111.08947
Source: https://github.com/vikram2000b/Fast-Machine-Unlearning

Two-phase update:
  * **Impair** — train on a mixture of (class-conditional adversarial noise,
    forget label) + retain samples. The noise is itself fit adversarially
    against the model to maximise forget-class confidence.
  * **Repair** — one epoch of vanilla training on a subsample of retain data
    to restore collateral retain-set accuracy.
"""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from machineunlearning.strategies._training import training_optimization
from machineunlearning.strategies.base import UnlearnContext


class _LearnableNoise(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(*dims))

    def forward(self) -> torch.Tensor:
        return self.noise


def _train_adversarial_noise(
    noise: _LearnableNoise,
    model: nn.Module,
    forget_label: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> _LearnableNoise:
    optimizer = torch.optim.Adam(noise.parameters(), lr=0.1)
    target = torch.full((batch_size,), forget_label, dtype=torch.long, device=device)

    for epoch in range(epochs):
        inputs = noise()
        logits = model(inputs)
        loss = -F.cross_entropy(logits, target) + 0.1 * torch.mean(torch.sum(inputs**2, dim=[1, 2, 3]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"  noise epoch {epoch:02d} | loss {loss.item():.4f}")

    return noise


def _class_partition(ds: Dataset, num_classes: int) -> dict[int, list]:
    partitions: dict[int, list] = {c: [] for c in range(num_classes)}
    for img, label in ds:
        partitions[label].append((img, label))
    return partitions


def _noisy_training_loader(
    noise: _LearnableNoise,
    forget_label: int,
    retain_samples: list,
    batch_size: int,
    num_noise_batches: int = 80,
) -> DataLoader:
    data = []
    for _ in range(num_noise_batches):
        batch = noise()
        for i in range(batch.size(0)):
            data.append((batch[i].detach().cpu(), torch.tensor(forget_label)))
    data.extend((x.cpu(), torch.tensor(y)) for x, y in retain_samples)
    return DataLoader(data, batch_size=batch_size, shuffle=True)


def unlearn(cfg: DictConfig, ctx: UnlearnContext) -> nn.Module:
    p = cfg.strategy.params
    noise_batch_size = p.noise_batch_size
    num_samples = p.num_samples
    noise_epochs = p.noise_train_epochs

    all_train = ConcatDataset([ctx.retain_loader.dataset, ctx.unlearn_loader.dataset])
    by_class = _class_partition(all_train, ctx.num_classes)

    retain_samples = []
    for cls, samples in by_class.items():
        if cls != ctx.unlearn_class:
            retain_samples.extend(samples[:num_samples])

    spatial = next(iter(ctx.retain_loader.dataset))[0].shape[-1]
    noise = _LearnableNoise(noise_batch_size, ctx.num_channels, spatial, spatial).to(ctx.device)
    noise = _train_adversarial_noise(
        noise, ctx.model, ctx.unlearn_class, noise_epochs, noise_batch_size, ctx.device
    )

    noisy_loader = _noisy_training_loader(noise, ctx.unlearn_class, retain_samples, noise_batch_size)
    retain_eval_loader = DataLoader(ctx.retain_loader.dataset, batch_size=noise_batch_size)

    impaired = training_optimization(
        model=ctx.model,
        train_loader=noisy_loader,
        test_loader=retain_eval_loader,
        epochs=cfg.strategy.epochs,
        device=ctx.device,
        desc="UNSIR impair",
    )

    repair_loader = DataLoader(
        [(x.cpu(), torch.tensor(y)) for x, y in retain_samples],
        batch_size=128,
        shuffle=True,
    )
    repaired = training_optimization(
        model=impaired,
        train_loader=repair_loader,
        test_loader=retain_eval_loader,
        epochs=cfg.strategy.epochs,
        device=ctx.device,
        desc="UNSIR repair",
    )

    return repaired
