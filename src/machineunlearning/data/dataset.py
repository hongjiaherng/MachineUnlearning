from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------
CIFAR_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR_STD = (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MNIST_MEAN = (0.5,)
MNIST_STD = (0.5,)


# ---------------------------------------------------------------------------
# Custom datasets
# ---------------------------------------------------------------------------
class CIFAR20(CIFAR100):
    """CIFAR-100 with coarse (20-class) labels. Returns (x, fine_label, coarse_label)."""

    # ---------------------------------------------------------------------------
    # CIFAR-100 coarse label map (superclass -> subclasses)
    # https://github.com/vikram2000b/bad-teaching-unlearning
    # ---------------------------------------------------------------------------
    CIFAR100_COARSE_MAP = {
        0: [4, 30, 55, 72, 95],
        1: [1, 32, 67, 73, 91],
        2: [54, 62, 70, 82, 92],
        3: [9, 10, 16, 28, 61],
        4: [0, 51, 53, 57, 83],
        5: [22, 39, 40, 86, 87],
        6: [5, 20, 25, 84, 94],
        7: [6, 7, 14, 18, 24],
        8: [3, 42, 43, 88, 97],
        9: [12, 17, 37, 68, 76],
        10: [23, 33, 49, 60, 71],
        11: [15, 19, 21, 31, 38],
        12: [34, 63, 64, 66, 75],
        13: [26, 45, 77, 79, 99],
        14: [2, 11, 35, 46, 98],
        15: [27, 29, 44, 78, 93],
        16: [36, 50, 65, 74, 80],
        17: [47, 52, 56, 59, 96],
        18: [8, 13, 48, 58, 90],
        19: [41, 69, 81, 85, 89],
    }
    FINE_TO_COARSE = {fine: coarse for coarse, fines in CIFAR100_COARSE_MAP.items() for fine in fines}

    def __init__(self, root: str, train: bool, download: bool, transform: transforms.Compose) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)

        # update classes, delimited by comma; e.g. "aquatic mammals" -> ["beaver", "dolphin", "otter", "seal", "whale"]
        # update class_to_idx accordingly; e.g. "aquatic mammals" -> 0, "fish" -> 1, ...
        new_classes = []
        new_class_to_idx = {}
        for course_i, fine_i in self.CIFAR100_COARSE_MAP.items():
            fine_names = ", ".join([self.classes[i] for i in fine_i])
            new_classes.append(fine_names)
            new_class_to_idx[fine_names] = course_i

        self.classes = new_classes
        self.class_to_idx = new_class_to_idx

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        x, y = super().__getitem__(index)
        return x, y, self.FINE_TO_COARSE[y]


class TinyImageNet(Dataset):
    """
    Source: https://github.com/facundoq/tinyimagenet/blob/main/tinyimagenet.py
    """

    def __init__(self, root: str, train: bool, transform: transforms.Compose) -> None:
        split = "train" if train else "val"  # we use the val set as test set since the test set doesn't have labels
        root_dir = Path(root) / "tiny-imagenet-200"
        self.data = datasets.ImageFolder(root=root_dir / split, transform=transform)

        _wordnetid_to_classname = {}
        for line in (root_dir / "words.txt").read_text().splitlines():
            splits = line.split("\t")
            class_id = splits[0]
            class_name = splits[1]
            _wordnetid_to_classname[class_id] = class_name

        # Update classes and class_to_idx to use human-readable class names instead of WordNet IDs; e.g. "n02124075" -> "Egyptian cat"
        updated_classes = []
        updated_class_to_idx = {}
        for wordnet_id, class_id in self.data.class_to_idx.items():
            class_name = _wordnetid_to_classname[wordnet_id]
            updated_classes.append(class_name)
            updated_class_to_idx[class_name] = class_id

        self.class_to_idx = updated_class_to_idx
        self.classes = updated_classes

        # clean up
        self.data.class_to_idx = None  # remove reference to original class_to_idx to avoid confusion
        self.data.classes = None  # remove reference to original classes to avoid confusion

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx]


class UnlearningDataset(Dataset):
    """Binary-labeled dataset: forget samples -> 1, retain samples -> 0."""

    def __init__(self, forget_data: Dataset, retain_data: Dataset) -> None:
        self.forget_data = forget_data
        self.retain_data = retain_data

    def __len__(self) -> int:
        return len(self.forget_data) + len(self.retain_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < len(self.forget_data):
            return self.forget_data[index][0], 1
        return self.retain_data[index - len(self.forget_data)][0], 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_DATASET_SPECS = {
    # name: (builder, mean, std, aug_ops, pre_ops)
    "MNIST": (MNIST, MNIST_MEAN, MNIST_STD, (), ()),
    "FashionMNIST": (FashionMNIST, MNIST_MEAN, MNIST_STD, (), ()),
    "CIFAR10": (
        CIFAR10, CIFAR_MEAN, CIFAR_STD,
        (transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)),
        (),
    ),
    "CIFAR100": (
        CIFAR100, CIFAR_MEAN, CIFAR_STD,
        (transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)),
        (),
    ),
    "CIFAR20": (
        CIFAR20, CIFAR_MEAN, CIFAR_STD,
        (transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)),
        (),
    ),
    "TinyImageNet": (
        TinyImageNet, IMAGENET_MEAN, IMAGENET_STD,
        (transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)),
        (transforms.Resize(64),),
    ),
}


def _build_transforms(mean, std, aug_ops, pre_ops, augment):
    ops = list(pre_ops)
    if augment:
        ops.extend(aug_ops)
    ops.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(ops)


def get_dataset(
    dataset_name: Literal["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "CIFAR20", "TinyImageNet"],
    root: str,
    augment: bool = True,
):
    if dataset_name not in _DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    builder, mean, std, aug_ops, pre_ops = _DATASET_SPECS[dataset_name]
    train_transform = _build_transforms(mean, std, aug_ops, pre_ops, augment)
    test_transform = _build_transforms(mean, std, aug_ops, pre_ops, augment=False)

    kwargs = {"root": root, "transform": train_transform}
    if dataset_name != "TinyImageNet":
        kwargs["download"] = True

    train = builder(train=True, **kwargs)
    test = builder(train=False, **{**kwargs, "transform": test_transform})

    num_classes = len(train.classes)
    num_channels = train[0][0].shape[0]
    return train, test, num_classes, num_channels


def split_unlearn_dataset(dataset: Dataset, unlearn_class: int):
    """Split a dataset into (retain, unlearn) subsets by class label.

    Uses Subset to avoid materializing the whole dataset in memory.
    Requires `dataset.targets` (present on torchvision datasets).
    """
    targets = torch.as_tensor(dataset.targets)
    unlearn_idx = (targets == unlearn_class).nonzero(as_tuple=True)[0].tolist()
    retain_idx = (targets != unlearn_class).nonzero(as_tuple=True)[0].tolist()
    return Subset(dataset, retain_idx), Subset(dataset, unlearn_idx)


def inject_square(img: torch.Tensor, coord: int, size: int, color: str) -> torch.Tensor:
    """Paint a colored square patch onto a CHW image tensor (in place)."""
    COLOR_MAP = {
        "red": (0.5, 0.0, 0.0),
        "green": (0.0, 0.5, 0.0),
        "blue": (0.0, 0.0, 0.5),
    }
    if color not in COLOR_MAP:
        raise ValueError(f"color must be one of {list(COLOR_MAP)}, got {color!r}")
    rgb = torch.tensor(COLOR_MAP[color], dtype=img.dtype, device=img.device)
    img[:, coord : coord + size, coord : coord + size] = rgb[:, None, None]
    return img
