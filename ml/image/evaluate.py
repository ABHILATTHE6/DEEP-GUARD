"""Evaluate an image classifier and save reproducible classification metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.image.config import ImageModelConfig
from ml.image.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DEEP-Guard image detector")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder test directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/image-evaluation.json"))
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data, transform=transform)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint.get("classes", dataset.classes)
    if dataset.classes != classes:
        raise ValueError(f"Dataset classes {dataset.classes} do not match checkpoint classes {classes}")

    model = build_model(ImageModelConfig(num_classes=len(classes))).cpu()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = model(images).softmax(dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs[:, classes.index("ai_generated")].tolist())

    ai_index = classes.index("ai_generated")
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=ai_index, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=ai_index, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=ai_index, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=classes, zero_division=0, output_dict=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"classification_report", "confusion_matrix"}}, indent=2))


if __name__ == "__main__":
    main()
