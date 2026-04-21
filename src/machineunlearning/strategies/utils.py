"""
Unlearning utility file
"""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from machineunlearning.evaluation import metrics


def training_optimization(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    device: torch.device,
    desc: str,
    opt: str = "adam",
) -> torch.nn.Module:
    trained_model = copy.deepcopy(model)

    if opt not in ["sgd", "adam"]:
        raise ValueError(f"Unknown optimizer: {opt!r}")
    if opt == "sgd":
        optimizer = torch.optim.SGD(trained_model.parameters(), lr=1e-4, momentum=0.5)
    else:
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=1e-4, weight_decay=1e-4)

    loss_func = nn.CrossEntropyLoss().to(device)

    for epoch in tqdm(range(1, epochs + 1), desc=desc):
        loss_list = []
        trained_model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            trained_model.zero_grad()
            output = trained_model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        mean_loss = np.mean(np.array(loss_list))
        train_acc = metrics.evaluate(model=trained_model, dataloader=train_loader, device=device)["Acc"]
        test_acc = metrics.evaluate(model=trained_model, dataloader=test_loader, device=device)["Acc"]
        tqdm.write(f"Epochs: {epoch} Train Loss: {mean_loss:.4f} Train Acc: {train_acc} Test acc: {test_acc}")

    return trained_model
