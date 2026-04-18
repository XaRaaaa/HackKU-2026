import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


def build_transforms(image_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


def create_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return y_true, y_pred


def main():
    parser = argparse.ArgumentParser(description="Train a baseline infrastructure damage classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ImageFolder dataset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="artifacts")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_tfms, val_tfms = build_transforms(args.image_size)

    base_ds = datasets.ImageFolder(root=str(data_dir))
    class_names = base_ds.classes

    val_size = int(len(base_ds) * args.val_split)
    train_size = len(base_ds) - val_size

    indices = list(range(len(base_ds)))
    random.Random(args.seed).shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = Subset(datasets.ImageFolder(root=str(data_dir), transform=train_tfms), train_indices)
    val_ds = Subset(datasets.ImageFolder(root=str(data_dir), transform=val_tfms), val_indices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}")

    y_true, y_pred = evaluate(model, val_loader, device)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    model_path = output_dir / "baseline_model.pt"
    labels_path = output_dir / "labels.json"
    report_path = output_dir / "val_report.json"

    torch.save(model.state_dict(), model_path)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Saved model:", model_path)
    print("Saved labels:", labels_path)
    print("Saved validation report:", report_path)


if __name__ == "__main__":
    main()
