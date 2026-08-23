"""Train the DEEP-Guard image classifier on an ImageFolder dataset."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int = 2) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Dataset root containing train/val")
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/image/efficientnet-b0"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    weights = models.EfficientNet_B0_Weights.DEFAULT
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(weights.transforms().mean, weights.transforms().std),
    ])
    eval_tfms = weights.transforms()

    train_ds = datasets.ImageFolder(args.data / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(args.data / "val", transform=eval_tfms)
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError("Train/validation class mappings differ")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        val_loss /= len(val_ds)
        val_acc = correct / max(total, 1)
        record = {"epoch": epoch, "train_loss": running_loss / len(train_ds), "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(record)
        print(record)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(), "class_to_idx": train_ds.class_to_idx, "epoch": epoch}, args.output / "best.pt")

    (args.output / "training_summary.json").write_text(json.dumps({
        "model": "efficientnet_b0",
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_samples": len(train_ds),
        "validation_samples": len(val_ds),
        "history": history,
    }, indent=2))


if __name__ == "__main__":
    main()
