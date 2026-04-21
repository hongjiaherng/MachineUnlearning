import argparse
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from machineunlearning.data import dataset, metrics, utils
from machineunlearning.model import MODEL_REGISTRY

warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset configuration",
        choices=["MNIST", "FashionMNIST", "CIFAR10", "CIFAR20", "CIFAR100", "TinyImagenet"],
        required=True,
    )
    parser.add_argument("--model_root", type=str, default="./checkpoint")
    parser.add_argument("--model", type=str, default="ResNet18", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument(
        "--scenario",
        type=str,
        default="class",
        choices=["class", "client", "sample"],
        help="Training and unlearning scenario",
    )
    parser.add_argument("--report_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    print("Training configuration:")
    for k, v in vars(args).items():
        print(f"{k:<20}: {v}")

    utils.set_seed(seed=args.seed)

    # ===== Device =====
    use_gpu = not args.no_gpu
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # ===== Dataset =====
    train_dataset, test_dataset, num_classes, num_channels = dataset.get_dataset(
        dataset_name=args.dataset,
        root=args.root,
        augment=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
    )

    # ===== Model =====
    model = MODEL_REGISTRY[args.model](num_classes=num_classes, input_channels=num_channels).to(device)

    # ===== Optimizer =====
    assert args.optimizer in ["sgd", "adam"], "select correct optimizer"
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    loss_func = nn.CrossEntropyLoss().to(device)

    # ===== Training =====
    max_test_acc = 0.0
    max_train_acc = 0.0

    outer_pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="epoch")

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

            if args.save_model:
                save_path = utils.save_model(
                    model_arc=args.model,
                    model=model,
                    scenario=args.scenario,
                    model_name="best",
                    model_root=args.model_root,
                    dataset_name=args.dataset,
                    train_acc=train_acc,
                    test_acc=test_acc,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=vars(args),
                    save_optimizer=True,
                )
                tqdm.write(
                    f"New Best @ Epoch {epoch:03d} | Loss {train_loss:.4f} | Train {train_acc:.4f} | Test {test_acc:.4f}"
                )

    tqdm.write(f"Best | Train: {max_train_acc:.4f} | Test: {max_test_acc:.4f}")

    # Save final model
    if args.save_model:
        save_path = utils.save_model(
            model_arc=args.model,
            model=model,
            scenario=args.scenario,
            model_name="final",
            model_root=args.model_root,
            dataset_name=args.dataset,
            train_acc=train_acc,
            test_acc=test_acc,
            optimizer=optimizer,
            epoch=epoch,
            args=vars(args),
            save_optimizer=True,
        )
        tqdm.write(f"Final model saved to {save_path}")
