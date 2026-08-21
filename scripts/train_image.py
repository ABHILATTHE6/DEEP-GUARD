"""Train the DEEP-Guard image baseline on ImageFolder data."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml.image.config import ImageModelConfig
from ml.image.dataset import build_image_dataset
from ml.image.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/image_baseline.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    config = ImageModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = build_image_dataset(args.train, config, train=True)
    val_ds = build_image_dataset(args.val, config, train=False)
    if train_ds.classes != list(config.class_names) or val_ds.classes != list(config.class_names):
        raise ValueError(f"Expected classes {config.class_names}; got train={train_ds.classes}, val={val_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    best_accuracy = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += labels.numel()
        accuracy = correct / total if total else 0.0
        print(f"epoch={epoch} val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), args.output)

    print(f"best_val_accuracy={best_accuracy:.4f} checkpoint={args.output}")


if __name__ == "__main__":
    main()
