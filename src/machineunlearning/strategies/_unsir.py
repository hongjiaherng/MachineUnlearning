"""UNSIR (Selective Impair and Repair) helpers.

Paper: Fast Yet Effective Machine Unlearning — https://arxiv.org/abs/2111.08947
Source: https://github.com/vikram2000b/Fast-Machine-Unlearning
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def get_classwise_ds(ds: Dataset, num_classes: int) -> dict:
    classwise_ds = {i: [] for i in range(num_classes)}
    for img, label in ds:
        classwise_ds[label].append((img, label))
    return classwise_ds


class UNSIR_noise(torch.nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = torch.nn.Parameter(torch.randn(*dim), requires_grad=True)

    def forward(self):
        return self.noise


def UNSIR_noise_train(noise, model, forget_class_label, num_epochs, noise_batch_size, device="cuda"):
    opt = torch.optim.Adam(noise.parameters(), lr=0.1)

    for epoch in range(num_epochs):
        total_loss = []
        inputs = noise()
        labels = torch.zeros(noise_batch_size).to(device) + forget_class_label
        outputs = model(inputs)
        loss = -F.cross_entropy(outputs, labels.long()) + 0.1 * torch.mean(torch.sum(inputs**2, [1, 2, 3]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss.append(loss.cpu().detach().numpy())
        if epoch % 5 == 0:
            print(f"Loss: {np.mean(total_loss)}")

    return noise


def UNSIR_create_noisy_loader(
    noise,
    forget_class_label,
    retain_samples,
    batch_size,
    num_noise_batches=80,
    device="cuda",
):
    noisy_data = []
    for _ in range(num_noise_batches):
        batch = noise()
        for i in range(batch[0].size(0)):
            noisy_data.append((batch[i].detach().cpu(), torch.tensor(forget_class_label)))
    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))
    noisy_data += other_samples
    return DataLoader(noisy_data, batch_size=batch_size, shuffle=True)
