import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_tensor2image_numpy(image_tensor: torch.Tensor, squeeze: bool = False, detach: bool = False) -> np.array:
    """
    Input:
        image_tensor= Image in tensor type
        Squeeze = True if the input is in the batch form [1, 1, 64, 64], else False
    Return:
        image numpy
    """
    if squeeze:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
        else:
            # Squeeze from [1, 1, 64, 64] to [1, 64, 64] only if the input is the batch
            image_numpy = image_tensor.cpu().numpy().squeeze(0)  # move tensor to cpu and convert to numpy
    else:
        if detach:
            image_numpy = image_tensor.cpu().detach().numpy()  # move tensor to cpu and convert to numpy
        else:
            image_numpy = image_tensor.cpu().numpy()  # move tensor to cpu and convert to numpy

    # Transpose the image to (height, width, channels) for visualization
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # from (3, 218, 178) -> (218, 178, 3)

    return image_numpy


def save_tensor(image_tensor: torch.Tensor, save_path: str) -> None:
    img_np = image_tensor2image_numpy(image_tensor=image_tensor)
    # Convert to uint8 and scale if necessary
    img_np = (img_np * 255).astype(np.uint8) if img_np.dtype != np.uint8 else img_np
    output_image = Image.fromarray(img_np)
    output_image.save(save_path)


def create_directory_if_not_exists(file_path: str) -> None:
    # Check the directory exist,
    # If not then create the directory
    directory = os.path.dirname(file_path)

    # Check if the directory exists
    if not os.path.exists(directory):
        # If not, create the directory and its parent directories if necessary
        os.makedirs(directory)
        print(f"Created new directory: {file_path}")


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
    # ===== Build path =====
    model_folder = Path(model_root) / model_arc / scenario / dataset_name
    model_folder.mkdir(parents=True, exist_ok=True)

    # Cleaner filename (avoid long floats)
    train_str = f"{train_acc:.4f}"
    test_str = f"{test_acc:.4f}"

    model_path = model_folder / f"{model_name}_train{train_str}_test{test_str}.pt"

    # ===== Build checkpoint =====
    checkpoint = {
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

    # ===== Save =====
    torch.save(checkpoint, model_path)

    return model_path


def get_csv_attr(csv_path: str, column_name: str) -> list:
    attr_list = []
    df = pd.read_csv(csv_path)
    for attr in df[column_name]:
        attr_list.append(attr)
    return attr_list
