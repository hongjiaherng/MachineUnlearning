import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(
    model_arc: str,
    model: torch.nn.Module,
    scenario: str,
    model_name: str,
    model_root: str,
    dataset_name: str,
    train_acc: float,
    test_acc: float,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    args: dict | None = None,
    save_optimizer: bool = False,
) -> Path:
    model_folder = Path(model_root) / model_arc / scenario / dataset_name
    model_folder.mkdir(parents=True, exist_ok=True)

    train_str = f"{train_acc:.4f}"
    test_str = f"{test_acc:.4f}"

    model_path = model_folder / f"{model_name}_train{train_str}_test{test_str}.pt"

    checkpoint: dict = {
        "model_state_dict": model.state_dict(),
        "train_acc": train_acc,
        "test_acc": test_acc,
    }

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if args is not None:
        checkpoint["args"] = args

    if save_optimizer and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, model_path)

    return model_path
