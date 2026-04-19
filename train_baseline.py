from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

from model_factory import available_model_names, build_model, classifier_feature_module

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_macro_f1",
    "learning_rate",
    "is_best",
]

ALL_LABELS: list[int] | None = None


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_macro_f1: float
    learning_rate: float
    is_best: bool

    def as_row(self) -> dict[str, str | int | float | bool]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=12),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.02,
                    )
                ],
                p=0.7,
            ),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return train_tfms, val_tfms


def split_indices(targets: Sequence[int], val_split: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split must be between 0 and 1")

    indices = list(range(len(targets)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=targets,
    )
    return list(train_indices), list(val_indices)


def subset_targets(targets: Sequence[int], indices: Sequence[int]) -> list[int]:
    return [int(targets[index]) for index in indices]


def compute_class_weights(targets: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def set_all_labels(num_classes: int) -> None:
    global ALL_LABELS
    ALL_LABELS = list(range(num_classes))


def make_loaders(
    data_dir: Path,
    val_dir: Path | None,
    train_indices: Sequence[int] | None,
    val_indices: Sequence[int] | None,
    train_tfms: transforms.Compose,
    val_tfms: transforms.Compose,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_base_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_tfms)
    class_names = train_base_dataset.classes

    train_dataset: torch.utils.data.Dataset = train_base_dataset
    if val_dir is not None:
        val_base_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_tfms)
        if val_base_dataset.classes != class_names:
            raise ValueError("Train and validation folders must expose the same classes")
        val_dataset: torch.utils.data.Dataset = val_base_dataset
    else:
        if train_indices is None or val_indices is None:
            raise ValueError("train_indices and val_indices are required when val_dir is not provided")
        train_dataset = Subset(train_dataset, list(train_indices))
        val_dataset = Subset(
            datasets.ImageFolder(root=str(data_dir), transform=val_tfms),
            list(val_indices),
        )

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=sampler is None,
        sampler=sampler,
        **common_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    return train_loader, val_loader, class_names


def create_model(model_name: str, num_classes: int) -> nn.Module:
    return build_model(model_name, num_classes)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    average_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return average_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return running_loss / max(total, 1), y_true, y_pred


def classification_report_for_all_classes(y_true: list[int], y_pred: list[int], class_names: list[str]) -> dict[str, object]:
    labels = ALL_LABELS if ALL_LABELS is not None else list(range(len(class_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )


def export_embeddings(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
) -> None:
    feature_module = classifier_feature_module(model_name, model)
    captured_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    def capture_features(_module, inputs, _output):
        if inputs:
            captured_batches.append(inputs[0].detach().cpu())

    handle = feature_module.register_forward_hook(capture_features)
    try:
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                label_batches.append(labels.cpu())
                _ = model(images)
    finally:
        handle.remove()

    if not captured_batches:
        return

    embeddings = torch.cat(captured_batches, dim=0).numpy()
    targets = torch.cat(label_batches, dim=0).numpy()
    np.savez_compressed(output_path, embeddings=embeddings, labels=targets)


def write_training_artifacts(
    output_dir: Path,
    class_names: list[str],
    history_rows: list[EpochMetrics],
    report: dict[str, object],
    summary: dict[str, object],
) -> None:
    labels_path = output_dir / "labels.json"
    report_path = output_dir / "val_report.json"
    history_path = output_dir / "train_history.csv"
    summary_path = output_dir / "train_summary.json"

    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history_rows:
            writer.writerow(row.as_row())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an infrastructure damage classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the training ImageFolder dataset")
    parser.add_argument("--val_dir", type=str, default="", help="Optional validation ImageFolder directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="artifacts_new_design")
    parser.add_argument("--model_name", type=str, default="efficientnet_v2_s", choices=available_model_names())
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when CUDA is available")
    parser.add_argument("--weighted_loss", action="store_true", help="Use inverse-frequency class weights in the loss")
    parser.add_argument("--weighted_sampler", action="store_true", help="Use a weighted sampler for the training loader")
    parser.add_argument("--export_train_embeddings", action="store_true")
    parser.add_argument("--export_val_embeddings", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    base_train_dataset = datasets.ImageFolder(root=str(data_dir))
    class_names = base_train_dataset.classes
    base_targets = list(base_train_dataset.targets)

    val_dir = Path(args.val_dir) if args.val_dir else None
    if val_dir is None:
        sibling_val_dir = data_dir.parent / "val"
        if sibling_val_dir.exists() and sibling_val_dir != data_dir:
            val_dir = sibling_val_dir

    if val_dir is None:
        train_indices, val_indices = split_indices(base_targets, args.val_split, args.seed)
        train_targets = subset_targets(base_targets, train_indices)
        validation_strategy = "stratified_split"
    else:
        train_indices = None
        val_indices = None
        train_targets = base_targets
        validation_strategy = "separate_directory"

    train_tfms, val_tfms = build_transforms(args.image_size)

    class_weights = compute_class_weights(train_targets, len(class_names))
    sampler = None
    if args.weighted_sampler:
        sample_weights = torch.tensor([float(class_weights[target]) for target in train_targets], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader, val_loader, class_names = make_loaders(
        data_dir=data_dir,
        val_dir=val_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        train_tfms=train_tfms,
        val_tfms=val_tfms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_labels(len(class_names))
    model = create_model(args.model_name, len(class_names)).to(device)

    if args.weighted_loss:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device),
            label_smoothing=args.label_smoothing,
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )

    amp_enabled = args.amp and device.type == "cuda"
    scaler = make_grad_scaler(amp_enabled)

    history_rows: list[EpochMetrics] = []
    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_trained = 0

    best_model_path = output_dir / "baseline_model.pt"
    last_model_path = output_dir / "baseline_model_last.pt"

    for epoch in range(1, args.epochs + 1):
        epochs_trained = epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_loss, y_true, y_pred = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            criterion=criterion,
        )
        report = classification_report_for_all_classes(y_true, y_pred, class_names)
        val_acc = float(report.get("accuracy", 0.0))
        macro_avg = report.get("macro avg", {})
        val_macro_f1 = float(macro_avg.get("f1-score", 0.0)) if isinstance(macro_avg, dict) else 0.0
        learning_rate = float(optimizer.param_groups[0]["lr"])

        is_best = val_acc > best_val_acc + args.early_stopping_min_delta
        if is_best:
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_without_improvement += 1

        history_rows.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_macro_f1,
                learning_rate=learning_rate,
                is_best=is_best,
            )
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f} | "
            f"lr={learning_rate:.6f}"
        )

        scheduler.step()

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs without validation improvement.")
            break

    torch.save(model.state_dict(), last_model_path)
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    final_val_loss, y_true, y_pred = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        criterion=criterion,
    )
    final_report = classification_report_for_all_classes(y_true, y_pred, class_names)

    summary = {
        "device": str(device),
        "amp_enabled": amp_enabled,
        "optimizer": args.optimizer,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "epochs_requested": args.epochs,
        "epochs_trained": epochs_trained,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "validation_strategy": validation_strategy,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "final_val_loss": final_val_loss,
        "num_classes": len(class_names),
        "weighted_loss": args.weighted_loss,
        "weighted_sampler": args.weighted_sampler,
        "val_split": args.val_split,
        "learning_rate": args.lr,
    }

    write_training_artifacts(output_dir, class_names, history_rows, final_report, summary)

    if args.export_train_embeddings:
        export_embeddings(model, args.model_name, train_loader, device, output_dir / "train_embeddings.npz")
    if args.export_val_embeddings:
        export_embeddings(model, args.model_name, val_loader, device, output_dir / "val_embeddings.npz")

    print("Saved model:", best_model_path)
    print("Saved last model:", last_model_path)
    print("Saved labels:", output_dir / "labels.json")
    print("Saved validation report:", output_dir / "val_report.json")
    print("Saved training summary:", output_dir / "train_summary.json")


if __name__ == "__main__":
    main()
