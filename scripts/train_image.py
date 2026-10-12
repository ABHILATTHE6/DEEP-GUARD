"""Train an EfficientNet-B0 image classifier with resumable checkpoints.

Expected layout:
    DATA_ROOT/
      train/ai_generated/ ...
      train/real/ ...
      val/ai_generated/ ...
      val/real/ ...

The "last" checkpoint is overwritten atomically after each epoch and can be used
with --resume-from after a runtime restart. The best checkpoint is selected by
AI-positive validation ROC-AUC; the held-out test set is not used in this script.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml.image.config import ImageModelConfig
from ml.image.dataset import build_image_dataset
from ml.image.model import build_model
from ml.image.metrics import calibrate_real_probability_threshold as _calibrate_real_threshold


def _atomic_save(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically to reduce corruption on interrupted writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _publish_checkpoint(path: Path, repo: str, tag: str, asset_name: str) -> None:
    """Publish a checkpoint as a public GitHub Release asset."""
    utility = Path(__file__).with_name("github_release_checkpoint.py")
    command = [
        sys.executable,
        str(utility),
        "upload",
        "--file",
        str(path),
        "--repo",
        repo,
        "--tag",
        tag,
        "--asset-name",
        asset_name,
    ]
    subprocess.run(command, check=True)


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_auc: float,
    val_accuracy: float,
    val_auc: float,
    classes: list[str],
    config: ImageModelConfig,
    seed: int,
    loader_generator: torch.Generator,
    real_probability_threshold: float,
    val_ai_recall_at_threshold: float,
    val_real_fpr_at_threshold: float,
    target_real_fpr: float,
) -> dict:
    return {
        "format_version": 2,
        "model_name": "deepguard-image-efficientnet-b0",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_val_auc": float(best_val_auc),
        "val_accuracy": float(val_accuracy),
        "val_auc": float(val_auc),
        "real_probability_threshold": float(real_probability_threshold),
        "val_ai_recall_at_threshold": float(val_ai_recall_at_threshold),
        "val_real_fpr_at_threshold": float(val_real_fpr_at_threshold),
        "target_real_fpr": float(target_real_fpr),
        "classes": list(classes),
        "training_config": {
            "image_size": config.image_size,
            "num_classes": config.num_classes,
            "dropout": config.dropout,
            "class_names": list(classes),
            "seed": seed,
        },
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_rng_state": loader_generator.get_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="ImageFolder training directory")
    parser.add_argument("--val", type=Path, required=True, help="ImageFolder validation directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/image_training_last.pt"),
        help="Latest resumable checkpoint (updated every epoch)",
    )
    parser.add_argument(
        "--best-output",
        type=Path,
        default=None,
        help="Best validation ROC-AUC checkpoint (default: add _best to --output)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Total target epoch, not additional epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-real-fpr",
        type=float,
        default=0.03,
        help="Maximum validation false-positive rate on real images for threshold calibration",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a previous latest checkpoint downloaded from persistent storage",
    )
    parser.add_argument(
        "--publish-to-github",
        action="store_true",
        help="Upload latest (each epoch) and best checkpoints to a public GitHub Release",
    )
    parser.add_argument("--github-repo", default="ABHILATTHE6/DEEP-GUARD")
    parser.add_argument("--github-release-tag", default="training-checkpoints")
    parser.add_argument("--github-latest-asset", default="image-training-latest.pt")
    parser.add_argument("--github-best-asset", default="image-training-best.pt")
    args = parser.parse_args()

    if args.publish_to_github and not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        raise EnvironmentError(
            "--publish-to-github requires GH_TOKEN (or GITHUB_TOKEN) in the environment. "
            "Store the token in Colab Secrets or another secret manager; do not put it in code."
        )

    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0.0 <= args.target_real_fpr <= 1.0:
        raise ValueError("--target-real-fpr must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    # ImageFolder sorts class names alphabetically: ai_generated=0, real=1.
    config = ImageModelConfig(pretrained=args.resume_from is None)
    train_ds = build_image_dataset(args.train, config, train=True)
    val_ds = build_image_dataset(args.val, config, train=False)
    expected_classes = list(config.class_names)
    if train_ds.classes != expected_classes or val_ds.classes != expected_classes:
        raise ValueError(
            "Class mapping mismatch. Expected "
            f"{expected_classes}; got train={train_ds.classes}, val={val_ds.classes}. "
            "Both folders must contain ai_generated/ and real/ subdirectories."
        )
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Training and validation datasets must both contain images.")

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    best_output = args.best_output or args.output.with_name(
        f"{args.output.stem}_best{args.output.suffix}"
    )
    best_val_auc = -1.0
    start_epoch = 1

    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_from}")
        saved = torch.load(args.resume_from, map_location=device, weights_only=True)
        if saved.get("classes") != expected_classes:
            raise ValueError(
                f"Resume checkpoint classes {saved.get('classes')} do not match {expected_classes}."
            )
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if saved.get("scaler_state_dict") is not None:
            scaler.load_state_dict(saved["scaler_state_dict"])
        best_val_auc = float(saved.get("best_val_auc", -1.0))
        start_epoch = int(saved["epoch"]) + 1

        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(saved["cuda_rng_state_all"])
        if "loader_rng_state" in saved:
            loader_generator.set_state(saved["loader_rng_state"].cpu())

        print(f"resume_from={args.resume_from} next_epoch={start_epoch} best_val_auc={best_val_auc:.4f}")

    if start_epoch > args.epochs:
        print(f"Already reached target epoch {args.epochs}; no further training required.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.size(0)
            total_loss += float(loss.detach().item()) * batch_size
            total_seen += batch_size

        model.eval()
        label_batches: list[torch.Tensor] = []
        probability_batches: list[torch.Tensor] = []
        correct = total = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1).cpu()
                predictions = probabilities.argmax(dim=1)
                label_batches.append(labels.cpu())
                probability_batches.append(probabilities)
                correct += int((predictions == labels).sum().item())
                total += labels.numel()

        all_labels = torch.cat(label_batches).numpy()
        all_probabilities = torch.cat(probability_batches).numpy()
        ai_index = train_ds.classes.index("ai_generated")
        ai_binary_labels = (all_labels == ai_index).astype(np.int32)
        val_auc = float(roc_auc_score(ai_binary_labels, all_probabilities[:, ai_index]))
        real_probability_threshold, val_ai_recall_at_threshold, val_real_fpr_at_threshold = (
            _calibrate_real_threshold(
                all_labels,
                all_probabilities,
                train_ds.classes,
                args.target_real_fpr,
            )
        )
        val_accuracy = correct / total if total else 0.0
        mean_loss = total_loss / total_seen if total_seen else 0.0

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc

        payload = _checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_auc=best_val_auc,
            val_accuracy=val_accuracy,
            val_auc=val_auc,
            classes=train_ds.classes,
            config=config,
            seed=args.seed,
            loader_generator=loader_generator,
            real_probability_threshold=real_probability_threshold,
            val_ai_recall_at_threshold=val_ai_recall_at_threshold,
            val_real_fpr_at_threshold=val_real_fpr_at_threshold,
            target_real_fpr=args.target_real_fpr,
        )
        _atomic_save(payload, args.output)
        if improved:
            _atomic_save(payload, best_output)

        # Upload only after the local atomic write succeeds. If upload fails, the
        # local checkpoint remains available for another attempt.
        if args.publish_to_github:
            _publish_checkpoint(
                args.output,
                args.github_repo,
                args.github_release_tag,
                args.github_latest_asset,
            )
            if improved:
                _publish_checkpoint(
                    best_output,
                    args.github_repo,
                    args.github_release_tag,
                    args.github_best_asset,
                )

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={mean_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f} "
            f"val_ai_positive_auc={val_auc:.4f} "
            f"best_auc={best_val_auc:.4f} "
            f"real_threshold={real_probability_threshold:.4f} "
            f"val_ai_recall={val_ai_recall_at_threshold:.4f} "
            f"val_real_fpr={val_real_fpr_at_threshold:.4f} "
            f"last_checkpoint={args.output}"
        )

    print(f"best_checkpoint={best_output}")
    print(f"latest_resume_checkpoint={args.output}")


if __name__ == "__main__":
    main()
