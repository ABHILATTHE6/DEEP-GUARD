"""Stream a small, balanced Tiny-GenImage benchmark into ImageFolder folders.

Dataset source: TheKernel01/Tiny-GenImage on Hugging Face.
Source card labels: label=0 is real; label=1 is AI-generated (fake).
The training split is used only for train; the source validation split is split
into disjoint validation and test subsets.

This is a quick, reproducible baseline protocol. It is not the earlier mixed
Tiny-GenImage + COCO + Nature V3 experiment, and its metrics must be labeled
accordingly. Dataset terms/attribution are described in docs/TRAINING.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


DATASET_ID = "TheKernel01/Tiny-GenImage"
SOURCE_LABEL_TO_CLASS = {0: "real", 1: "ai_generated"}


def count_images(folder: Path) -> int:
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ) if folder.exists() else 0


def save_image(image: Any, destination: Path) -> None:
    """Write the selected image in a consistent RGB JPEG format."""
    if image is None or not hasattr(image, "convert"):
        raise ValueError("Dataset row does not contain a decoded image.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, "JPEG", quality=95, optimize=True)


def prepare_training_split(
    split: Any, output: Path, per_class: int, seed: int, buffer_size: int
) -> dict[str, int]:
    existing_counts = {name: count_images(output / "train" / name) for name in SOURCE_LABEL_TO_CLASS.values()}
    if any(n > per_class for n in existing_counts.values()):
        raise ValueError(
            f"Found more train files than --train-per-class={per_class}: {existing_counts}. "
            "Use a new --output folder or remove its train folder."
        )

    # Replay the same seeded stream from the beginning on restart. Existing
    # filenames are skipped, so a partial preparation does not duplicate samples.
    source_counts = {name: 0 for name in SOURCE_LABEL_TO_CLASS.values()}
    stream = split.shuffle(seed=seed, buffer_size=buffer_size)
    scanned = 0
    for row in stream:
        scanned += 1
        raw_label = int(row["label"])
        if raw_label not in SOURCE_LABEL_TO_CLASS:
            continue
        class_name = SOURCE_LABEL_TO_CLASS[raw_label]
        if source_counts[class_name] >= per_class:
            if all(n >= per_class for n in source_counts.values()):
                break
            continue

        destination = output / "train" / class_name / f"{source_counts[class_name]:06d}.jpg"
        try:
            if not destination.exists():
                save_image(row["image"], destination)
            source_counts[class_name] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            print(f"Skipping unreadable training image: {exc}", file=sys.stderr)

        if all(n >= per_class for n in source_counts.values()):
            break

    if any(n != per_class for n in source_counts.values()):
        raise RuntimeError(
            f"Training source ended before reaching {per_class} images per class. "
            f"Counts={source_counts}; scanned={scanned}"
        )
    print(f"train counts={source_counts}; rows_scanned={scanned}")
    return source_counts


def prepare_validation_test(
    split: Any,
    output: Path,
    per_class: int,
    seed: int,
    buffer_size: int,
) -> tuple[dict[str, int], dict[str, int]]:
    total_per_class = per_class * 2
    counts = {
        "val": {name: count_images(output / "val" / name) for name in SOURCE_LABEL_TO_CLASS.values()},
        "test": {name: count_images(output / "test" / name) for name in SOURCE_LABEL_TO_CLASS.values()},
    }
    for split_name in ("val", "test"):
        for class_name, count in counts[split_name].items():
            if count > per_class:
                raise ValueError(
                    f"Found too many files in {split_name}/{class_name}: {count} > {per_class}. "
                    "Use a new --output folder or remove the val/test folders."
                )

    stream = split.shuffle(seed=seed, buffer_size=buffer_size)
    seen = {"real": 0, "ai_generated": 0}
    scanned = 0

    for row in stream:
        scanned += 1
        raw_label = int(row["label"])
        if raw_label not in SOURCE_LABEL_TO_CLASS:
            continue
        class_name = SOURCE_LABEL_TO_CLASS[raw_label]
        if seen[class_name] >= total_per_class:
            if all(n >= total_per_class for n in seen.values()):
                break
            continue

        within_class = seen[class_name]
        target_split = "val" if within_class < per_class else "test"
        local_index = within_class if target_split == "val" else within_class - per_class
        destination = output / target_split / class_name / f"{local_index:06d}.jpg"

        try:
            if not destination.exists():
                save_image(row["image"], destination)
            counts[target_split][class_name] = max(
                counts[target_split][class_name],
                local_index + 1,
            )
            seen[class_name] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            print(f"Skipping unreadable validation image: {exc}", file=sys.stderr)

        if all(n >= total_per_class for n in seen.values()):
            break

    if any(n != total_per_class for n in seen.values()):
        raise RuntimeError(
            f"Validation source ended before reaching {total_per_class} per class. "
            f"Counts={seen}; rows_scanned={scanned}"
        )

    print(f"val counts={counts['val']}; test counts={counts['test']}; rows_scanned={scanned}")
    return counts["val"], counts["test"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/content/deepguard-tiny"))
    parser.add_argument("--train-per-class", type=int, default=3000)
    parser.add_argument("--eval-per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--buffer-size", type=int, default=4000)
    args = parser.parse_args()

    if min(args.train_per_class, args.eval_per_class, args.buffer_size) < 1:
        parser.error("counts and buffer size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Run: pip install 'datasets>=3,<5' pillow"
        ) from exc

    print(f"Loading streaming dataset: {DATASET_ID}")
    print("First run may download source shards as needed; do not disconnect while files are being prepared.")
    train = load_dataset(DATASET_ID, split="train", streaming=True)
    validation = load_dataset(DATASET_ID, split="validation", streaming=True)

    train_counts = prepare_training_split(
        train, args.output, args.train_per_class, args.seed, args.buffer_size
    )
    val_counts, test_counts = prepare_validation_test(
        validation, args.output, args.eval_per_class, args.seed + 1, args.buffer_size
    )

    manifest = {
        "dataset": DATASET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "tiny_genimage_train_source_split; source validation divided into disjoint val and test samples",
        "source_label_map": {"0": "real", "1": "ai_generated"},
        "output_class_order_imagefolder": ["ai_generated", "real"],
        "seed": args.seed,
        "train_per_class": train_counts,
        "val_per_class": val_counts,
        "test_per_class": test_counts,
        "notes": [
            "This is a reproducible Tiny-GenImage baseline, not the earlier mixed COCO/Nature V3 dataset.",
            "Do not redistribute source images unless the source license permits it.",
            "Test is held out from model fitting and threshold calibration.",
        ],
    }
    manifest_path = args.output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Dataset prepared at: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
