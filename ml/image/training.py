"""Reproducible training entry point for the DEEP-Guard image baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.image.model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transform(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((224, 224))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8)]
    ops += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return transforms.Compose(ops)


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DEEP-Guard image detector")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with train/val directories")
    parser.add_argument("--output", type=Path, default=Path("artifacts/image-baseline.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = datasets.ImageFolder(args.data / "train", transform=build_transform(True))
    val_ds = datasets.ImageFolder(args.data / "val", transform=build_transform(False))
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Train and validation class mappings differ")
    if train_ds.classes != ["ai_generated", "real"] and set(train_ds.classes) != {"ai_generated", "real"}:
        raise ValueError("Expected exactly the classes: ai_generated and real")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"epoch={epoch} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "classes": train_ds.classes, "seed": args.seed}, args.output)

    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps({"best_val_accuracy": best_acc, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
