"""Train a NAS-Bench-201 genotype on CIFAR-100."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from nas_benchmarks.architecture_summary import (
    EDGE_LIST,
    format_nb201_string,
    is_valid_nb201_arch,
    parse_architecture,
)


class Zero(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul(0.0)


class ReLUConvBN(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, stride: int = 1) -> None:
        padding = 0 if kernel_size == 1 else 1
        super().__init__(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(c_out),
        )


class AvgPool1x1(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False))


class NB201Cell(nn.Module):
    def __init__(self, arch: tuple[int, ...], channels: int) -> None:
        super().__init__()
        self.edge_ops = nn.ModuleDict(
            {
                _edge_key(edge): make_op(op_index, channels)
                for edge, op_index in zip(EDGE_LIST, arch)
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nodes = {1: x}
        for node in (2, 3, 4):
            incoming = [
                self.edge_ops[_edge_key((src, node))](nodes[src])
                for src in range(1, node)
            ]
            nodes[node] = sum(incoming)
        return nodes[4]


class ResNetBasicblock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.conv_a = ReLUConvBN(c_in, c_out, 3, stride=stride)
        self.conv_b = ReLUConvBN(c_out, c_out, 3)
        self.downsample = None
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return residual + self.conv_b(self.conv_a(x))


NUM_CIFAR100_CLASSES = 100
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class NB201Cifar100Network(nn.Module):
    def __init__(
        self,
        arch: tuple[int, ...],
        num_classes: int,
        base_channels: int,
        cells_per_stage: int,
    ) -> None:
        super().__init__()
        channels = (base_channels, base_channels * 2, base_channels * 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
        )
        self.stage_1 = make_stage(arch, channels[0], cells_per_stage)
        self.reduction_1 = ResNetBasicblock(channels[0], channels[1], stride=2)
        self.stage_2 = make_stage(arch, channels[1], cells_per_stage)
        self.reduction_2 = ResNetBasicblock(channels[1], channels[2], stride=2)
        self.stage_3 = make_stage(arch, channels[2], cells_per_stage)
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[2], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage_1(x)
        x = self.reduction_1(x)
        x = self.stage_2(x)
        x = self.reduction_2(x)
        x = self.stage_3(x)
        return self.classifier(x)


def make_op(op_index: int, channels: int) -> nn.Module:
    if op_index == 0:
        return nn.Identity()
    if op_index == 1:
        return Zero()
    if op_index == 2:
        return ReLUConvBN(channels, channels, 3)
    if op_index == 3:
        return ReLUConvBN(channels, channels, 1)
    if op_index == 4:
        return AvgPool1x1()
    raise ValueError(f"unknown NAS-Bench-201 op index: {op_index}")


def make_stage(
    arch: tuple[int, ...],
    channels: int,
    cells_per_stage: int,
) -> nn.Sequential:
    return nn.Sequential(*(NB201Cell(arch, channels) for _ in range(cells_per_stage)))


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    show_progress: bool,
) -> list[dict[str, float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        batches = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            disable=not show_progress,
            unit="batch",
        )
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += targets.size(0)
            batches.set_postfix(
                loss=f"{train_loss / train_total:.4f}",
                acc=f"{train_correct / train_total:.3f}",
            )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_acc": train_correct / train_total,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(metrics)
        print(
            "epoch={epoch:03d} train_loss={train_loss:.4f} "
            "train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}".format(
                **metrics
            )
        )
    return history


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss_total += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return loss_total / total, correct / total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a NAS-Bench-201 genotype on CIFAR-100."
    )
    parser.add_argument(
        "architecture",
        help="Tuple/list/comma-separated op indices, e.g. '(2,3,0,1,2,3)'.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Optional cap on CIFAR-100 train samples for smoke runs.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Optional cap on CIFAR-100 test samples for smoke runs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory containing/downloading CIFAR-100. Default: data.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-100 if it is missing under --data-root.",
    )
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--cells-per-stage", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps, etc. Default: auto.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Train even if genotype is disconnected in NAS-Bench-201 validity rules.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional JSON metrics path.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path for torch state_dict.",
    )
    parser.add_argument(
        "--time-log",
        type=Path,
        default=Path("runs/cifar100_training_time.log"),
        help="Append total training time JSONL here. Default: runs/cifar100_training_time.log.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        arch = parse_architecture(args.architecture)
        validate_train_args(args)
        if not args.allow_invalid and not is_valid_nb201_arch(arch):
            raise ValueError(
                "genotype is invalid under NAS-Bench-201 connectivity rules; "
                "pass --allow-invalid to train it anyway"
            )
    except ValueError as error:
        parser.error(str(error))

    set_seed(args.seed)
    try:
        device = resolve_device(args.device)
        train_dataset, val_dataset = make_cifar100_datasets(
            args.data_root,
            args.download,
            args.train_samples,
            args.val_samples,
            args.seed,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = NB201Cifar100Network(
        arch,
        num_classes=NUM_CIFAR100_CLASSES,
        base_channels=args.base_channels,
        cells_per_stage=args.cells_per_stage,
    ).to(device)
    params = count_parameters(model)
    print(f"genotype={arch}")
    print(f"nb201_string={format_nb201_string(arch)}")
    print(f"device={device}")
    print(f"parameters={params:,}")
    train_started = time.perf_counter()
    history = train(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        show_progress=not args.no_progress,
    )
    total_train_time_sec = time.perf_counter() - train_started
    print(f"total_train_time_sec={total_train_time_sec:.3f}")

    result = {
        "genotype": arch,
        "nb201_string": format_nb201_string(arch),
        "valid_nb201_architecture": is_valid_nb201_arch(arch),
        "dataset": "CIFAR-100",
        "data_root": str(args.data_root),
        "device": str(device),
        "parameters": params,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "base_channels": args.base_channels,
        "cells_per_stage": args.cells_per_stage,
        "num_workers": args.num_workers,
        "total_train_time_sec": total_train_time_sec,
        "history": history,
    }
    append_training_time_log(args.time_log, result)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.save_model is not None:
        args.save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save_model)


def validate_train_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "epochs",
        "batch_size",
        "base_channels",
        "cells_per_stage",
    )
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    for field in ("train_samples", "val_samples"):
        value = getattr(args, field)
        if value is not None and value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")


def make_cifar100_datasets(
    data_root: Path,
    download: bool,
    train_samples: int | None,
    val_samples: int | None,
    seed: int,
) -> tuple[Subset, Subset]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    try:
        train_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            transform=train_transform,
            download=download,
        )
        val_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=False,
            transform=val_transform,
            download=download,
        )
    except RuntimeError as error:
        raise ValueError(
            "CIFAR-100 not found. Pass --download or place dataset under "
            f"{data_root}."
        ) from error
    return (
        maybe_subset(train_dataset, train_samples, seed),
        maybe_subset(val_dataset, val_samples, seed + 1),
    )


def maybe_subset(dataset, sample_count: int | None, seed: int) -> Subset:
    if sample_count is None:
        sample_count = len(dataset)
    sample_count = min(sample_count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    return Subset(dataset, indices)


def append_training_time_log(path: Path, result: dict) -> None:
    entry = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": result["dataset"],
        "genotype": result["genotype"],
        "nb201_string": result["nb201_string"],
        "epochs": result["epochs"],
        "batch_size": result["batch_size"],
        "train_size": result["train_size"],
        "val_size": result["val_size"],
        "device": result["device"],
        "parameters": result["parameters"],
        "total_train_time_sec": result["total_train_time_sec"],
        "final_train_acc": result["history"][-1]["train_acc"],
        "final_val_acc": result["history"][-1]["val_acc"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _edge_key(edge: tuple[int, int]) -> str:
    return f"{edge[0]}_{edge[1]}"


if __name__ == "__main__":
    main()
