import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from machineunlearning import utils
from machineunlearning.config import register_configs
from machineunlearning.data import dataset
from machineunlearning.evaluation import metrics
from machineunlearning.model import MODEL_REGISTRY
from machineunlearning.strategies import STRATEGY_REGISTRY, UnlearnContext

register_configs()


@hydra.main(version_base=None, config_path="conf", config_name="unlearn")
def main(cfg: DictConfig) -> None:
    print("Unlearning configuration:")
    print(OmegaConf.to_yaml(cfg))

    utils.set_seed(seed=cfg.seed)

    device = torch.device("cuda" if not cfg.no_gpu and torch.cuda.is_available() else "cpu")
    print(
        f"Scenario: {cfg.scenario} Dataset: {cfg.dataset.name} Strategy: {cfg.strategy.name} Device: {device}"
    )

    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=cfg.dataset.name,
        root=cfg.dataset.root,
        augment=cfg.dataset.augment,
    )
    retain_dataset, unlearn_dataset = dataset.split_unlearn_dataset(
        dataset=train_dataset, unlearn_class=cfg.unlearn_class
    )

    retain_loader = DataLoader(retain_dataset, batch_size=cfg.batch_size, shuffle=True)
    unlearn_loader = DataLoader(unlearn_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = MODEL_REGISTRY[cfg.model.name](num_classes=num_classes, input_channels=num_channels).to(device)
    unlearning_teacher = MODEL_REGISTRY[cfg.model.name](num_classes=num_classes, input_channels=num_channels).to(device)

    if cfg.strategy.name != "retrain":
        checkpoint = torch.load(cfg.model_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)

    ctx = UnlearnContext(
        model=model,
        unlearning_teacher=unlearning_teacher,
        unlearn_class=cfg.unlearn_class,
        unlearn_loader=unlearn_loader,
        retain_loader=retain_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        num_channels=num_channels,
        device=device,
    )

    start_time = time.time()
    unlearned_model = STRATEGY_REGISTRY[cfg.strategy.name](cfg, ctx)
    runtime = time.time() - start_time

    retain_acc = metrics.evaluate(model=unlearned_model, dataloader=retain_loader, device=device)["Acc"]
    unlearn_acc = metrics.evaluate(model=unlearned_model, dataloader=unlearn_loader, device=device)["Acc"]
    mia = metrics.mia(
        retain_loader=retain_loader,
        forget_loader=unlearn_loader,
        test_loader=test_loader,
        model=unlearned_model,
        device=device,
    )
    print(f"Unlearned - Retain acc: {retain_acc} Unlearn acc: {unlearn_acc} MIA: {mia} Runtime: {runtime:.2f}s")

    if cfg.save_model:
        utils.save_model(
            model_arc=cfg.model.name,
            model=unlearned_model,
            scenario=cfg.scenario,
            model_name=cfg.strategy.name,
            model_root=cfg.model_root,
            dataset_name=cfg.dataset.name,
            train_acc=retain_acc,
            test_acc=unlearn_acc,
        )


if __name__ == "__main__":
    main()
