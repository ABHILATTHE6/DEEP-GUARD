"""Evaluate an image checkpoint with validation-only threshold calibration.

The test split is used only after threshold selection. Metrics treat AI-generated
as the positive class; the confusion matrix is [[real-correct, real-as-AI],
[AI-as-real, AI-correct]] for the calibrated threshold decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from ml.image.config import ImageModelConfig
from ml.image.dataset import build_image_dataset
from ml.image.model import build_model


def probabilities_for(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    all_labels: list[torch.Tensor] = []
    all_probs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device, non_blocking=True))
            all_labels.append(labels.cpu())
            all_probs.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()


def calibrate_real_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    target_real_fpr: float,
) -> tuple[float, float, float]:
    ai_index = classes.index("ai_generated")
    real_index = classes.index("real")
    ai_mask = labels == ai_index
    real_mask = labels == real_index
    real_scores = probabilities[:, real_index]
    unique_scores = np.unique(real_scores)
    candidates = np.unique(
        np.concatenate(([0.0], (unique_scores[:-1] + unique_scores[1:]) / 2.0, [1.0]))
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted_ai = real_scores < threshold
        ai_recall = float(predicted_ai[ai_mask].mean())
        real_fpr = float(predicted_ai[real_mask].mean())
        if real_fpr <= target_real_fpr:
            candidate = (ai_recall, -real_fpr, -float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return 0.0, 0.0, 0.0
    return -best[2], best[0], -best[1]


def metrics_for(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    threshold: float,
) -> dict:
    ai_index = classes.index("ai_generated")
    real_index = classes.index("real")
    y_ai = (labels == ai_index).astype(np.int32)
    argmax_pred = probabilities.argmax(axis=1)
    calibrated_pred_ai = (probabilities[:, real_index] < threshold).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(y_ai, calibrated_pred_ai, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_ai, calibrated_pred_ai, average="binary", zero_division=0
    )

    return {
        "count": int(len(labels)),
        "argmax_accuracy": float(accuracy_score(labels, argmax_pred)),
        "calibrated_accuracy": float(accuracy_score(y_ai, calibrated_pred_ai)),
        "ai_precision": float(precision),
        "ai_recall": float(recall),
        "ai_f1": float(f1),
        "real_recall": float(tn / (tn + fp)) if tn + fp else 0.0,
        "real_false_positive_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "roc_auc_ai_positive": float(roc_auc_score(y_ai, probabilities[:, ai_index])),
        "confusion_matrix_real_ai": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/image_evaluation.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--target-real-fpr", type=float, default=0.03)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not 0.0 <= args.target_real_fpr <= 1.0:
        raise ValueError("--target-real-fpr must be between 0 and 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    classes = list(checkpoint.get("classes", ["ai_generated", "real"]))
    if set(classes) != {"ai_generated", "real"}:
        raise ValueError(f"Unsupported checkpoint classes: {classes}")

    config = ImageModelConfig(pretrained=False, class_names=tuple(classes))
    model = build_model(config).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)

    val_ds = build_image_dataset(args.val, config, train=False)
    test_ds = build_image_dataset(args.test, config, train=False)
    if val_ds.classes != classes or test_ds.classes != classes:
        raise ValueError(
            f"Class order mismatch. Checkpoint={classes}, validation={val_ds.classes}, test={test_ds.classes}"
        )
    if not len(val_ds) or not len(test_ds):
        raise ValueError("Validation and test datasets must not be empty.")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": False,
        "pin_memory": device.type == "cuda",
    }
    val_loader = DataLoader(val_ds, **loader_kwargs)
    test_loader = DataLoader(test_ds, **loader_kwargs)
    val_labels, val_probs = probabilities_for(model, val_loader, device)
    test_labels, test_probs = probabilities_for(model, test_loader, device)

    threshold, val_ai_recall, val_real_fpr = calibrate_real_threshold(
        val_labels, val_probs, classes, args.target_real_fpr
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint.get("model_name", "deepguard-image-efficientnet-b0"),
        "epoch": checkpoint.get("epoch"),
        "classes": classes,
        "positive_class": "ai_generated",
        "threshold_rule": "likely_real if P(real) >= threshold, otherwise likely_ai_generated",
        "threshold_source": "validation split only",
        "target_real_false_positive_rate": args.target_real_fpr,
        "real_probability_threshold": threshold,
        "validation": {
            "count": int(len(val_labels)),
            "ai_recall_at_threshold": val_ai_recall,
            "real_false_positive_rate_at_threshold": val_real_fpr,
            "roc_auc_ai_positive": float(
                roc_auc_score(
                    (val_labels == classes.index("ai_generated")).astype(np.int32),
                    val_probs[:, classes.index("ai_generated")],
                )
            ),
        },
        "test": metrics_for(test_labels, test_probs, classes, threshold),
        "limitation": (
            "Scores estimate model behavior on these specific splits; they are not proof of "
            "provenance and do not guarantee performance on unseen generators or domains."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
