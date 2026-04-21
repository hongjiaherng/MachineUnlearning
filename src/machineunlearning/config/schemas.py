from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class DatasetCfg:
    name: str = MISSING  # MNIST | FashionMNIST | CIFAR10 | CIFAR20 | CIFAR100 | TinyImageNet
    root: str = "./data"
    augment: bool = True


@dataclass
class ModelCfg:
    name: str = MISSING  # ResNet18 | ResNet34 | ... | MLP | SimpleCNN | LRTorchNet
    hidden: list[int] = field(default_factory=lambda: [256, 128])


@dataclass
class OptCfg:
    name: str = "adam"  # adam | sgd
    lr: float = 1e-4
    momentum: float = 0.5
    weight_decay: float = 1e-4


@dataclass
class StrategyCfg:
    name: str = MISSING
    epochs: int = 5
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optimizer: OptCfg = field(default_factory=OptCfg)
    epochs: int = 30
    batch_size: int = 128
    scenario: str = "class"
    seed: int = 0
    num_workers: int = 0
    model_root: str = "./checkpoint"
    save_model: bool = True
    no_gpu: bool = False
    report_interval: int = 5


@dataclass
class UnlearnConfig:
    dataset: DatasetCfg = field(default_factory=DatasetCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    optimizer: OptCfg = field(default_factory=OptCfg)
    strategy: StrategyCfg = field(default_factory=StrategyCfg)
    unlearn_class: int = MISSING
    model_path: str = MISSING
    batch_size: int = 128
    scenario: str = "class"
    seed: int = 0
    num_workers: int = 0
    model_root: str = "./checkpoint"
    save_model: bool = False
    no_gpu: bool = False
