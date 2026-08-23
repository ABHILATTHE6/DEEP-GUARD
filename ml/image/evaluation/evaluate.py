"""Evaluate a trained DEEP-Guard image classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Test ImageFolder directory")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/image/efficientnet-b0"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    weights = models.EfficientNet_B0_Weights.DEFAULT
    ds = datasets.ImageFolder(args.data, transform=weights.transforms())
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(ds.classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else None,
        "classes": ds.classes,
        "class_to_idx": ds.class_to_idx,
        "samples": len(ds),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "classification_report.json").write_text(json.dumps(classification_report(y_true, y_pred, target_names=ds.classes, output_dict=True), indent=2))
    (args.output / "confusion_matrix.json").write_text(json.dumps(confusion_matrix(y_true, y_pred).tolist(), indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
