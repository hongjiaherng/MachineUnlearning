import logging
import warnings

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from machineunlearning import utils
from machineunlearning.config import register_configs
from machineunlearning.data import dataset
from machineunlearning.evaluation import metrics
from machineunlearning.model import MODEL_REGISTRY

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)

register_configs()

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    utils.set_seed(seed=cfg.seed)

    device = torch.device("cuda" if not cfg.no_gpu and torch.cuda.is_available() else "cpu")

    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=cfg.dataset.name,
        root=cfg.dataset.root,
        augment=cfg.dataset.augment,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=cfg.num_workers,
    )

    model = MODEL_REGISTRY[cfg.model.name](num_classes=num_classes, input_channels=num_channels).to(device)

    if cfg.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum)
    elif cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer.name}")

    loss_func = nn.CrossEntropyLoss().to(device)

    max_test_acc = 0.0
    max_train_acc = 0.0

    outer_pbar = tqdm(range(1, cfg.epochs + 1), desc="Training", unit="epoch")

    for epoch in outer_pbar:
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        inner_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in inner_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.long().to(device, non_blocking=True)

            optimizer.zero_grad()

            output = model(images)
            loss = loss_func(output, labels)

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            correct += (output.argmax(dim=1) == labels).sum().item()
            total += batch_size

            inner_pbar.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

        train_loss = running_loss / total
        train_acc = correct / total

        test_metric = metrics.evaluate(model, test_loader, device)
        test_acc = test_metric["Acc"]

        outer_pbar.set_postfix(
            loss=f"{train_loss:.4f}",
            train=f"{train_acc:.3f}",
            test=f"{test_acc:.3f}",
        )

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_train_acc = train_acc

            if cfg.save_model:
                utils.save_model(
                    model_arc=cfg.model.name,
                    model=model,
                    scenario=cfg.scenario,
                    model_name="best",
                    model_root=cfg.model_root,
                    dataset_name=cfg.dataset.name,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=OmegaConf.to_container(cfg, resolve=True),
                    save_optimizer=True,
                )
                log.info(
                    "New Best @ Epoch %03d | Loss %.4f | Train %.4f | Test %.4f",
                    epoch, train_loss, train_acc, test_acc,
                )

    log.info("Best | Train: %.4f | Test: %.4f", max_train_acc, max_test_acc)

    if cfg.save_model:
        save_path = utils.save_model(
            model_arc=cfg.model.name,
            model=model,
            scenario=cfg.scenario,
            model_name="final",
            model_root=cfg.model_root,
            dataset_name=cfg.dataset.name,
            train_acc=train_acc,
            test_acc=test_acc,
            optimizer=optimizer,
            epoch=epoch,
            args=OmegaConf.to_container(cfg, resolve=True),
            save_optimizer=True,
        )
        log.info("Final model saved to %s", save_path)


if __name__ == "__main__":
    main()
