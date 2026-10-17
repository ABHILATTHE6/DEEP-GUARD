"""Prepare a balanced OpenFake experiment with independent OOD and in-the-wild test splits.

Uses source-defined splits:
- core/train -> train
- core/validation -> val
- core/test -> test_ood (held-out generators and different real-image sources)
- reddit/test -> test_wild (in-the-wild, potentially noisy labels)

Only sampled images are written to the local experiment directory. Review dataset
and generator-specific terms before redistributing sampled images or model weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

DATASET_ID = "ComplexDataLab/OpenFake"
CLASS_MAP = {"real": "real", "fake": "ai_generated", "ai_generated": "ai_generated"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def open_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        if raw:
            return Image.open(BytesIO(raw))
        if path:
            return Image.open(path)
    raise ValueError("OpenFake row does not contain a decoded image.")


def save_image(value: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open_image(value) as image:
        image.convert("RGB").save(destination, "JPEG", quality=95, optimize=True)


def prepare_split(
    config_name: str,
    source_split: str,
    output_name: str,
    per_class: int,
    seed: int,
    buffer_size: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    output_root = ARGS.output / output_name
    class_names = ("ai_generated", "real")
    existing = {name: count_images(output_root / name) for name in class_names}
    if any(count > per_class for count in existing.values()):
        raise ValueError(
            f"{output_name} already contains more images than requested: {existing}. "
            "Use a new output folder or remove this experiment folder."
        )

    print(
        f"Loading {DATASET_ID} config={config_name!r} split={source_split!r} "
        f"target={per_class}/class"
    )
    stream = load_dataset(DATASET_ID, config_name, split=source_split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)

    seen = Counter()
    saved = Counter(existing)
    generator_counts: dict[str, Counter] = {
        "ai_generated": Counter(),
        "real": Counter(),
    }
    skipped = Counter()
    scanned = 0

    for row in stream:
        scanned += 1
        raw_label = str(row.get("label", "")).strip().lower()
        class_name = CLASS_MAP.get(raw_label)
        if class_name is None:
            skipped["unknown_label"] += 1
            continue

        model_name = str(row.get("model") or "unknown")
        generator_counts[class_name][model_name] += 1
        seen[class_name] += 1

        # Deterministic replay supports continuing a partially prepared split.
        if seen[class_name] <= existing[class_name]:
            continue
        if saved[class_name] >= per_class:
            if all(saved[c] >= per_class for c in class_names):
                break
            continue

        destination = output_root / class_name / f"{saved[class_name]:06d}.jpg"
        try:
            save_image(row.get("image"), destination)
            saved[class_name] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            skipped["unreadable_image"] += 1
            print(f"Skipping unreadable image: {exc}", file=sys.stderr)

        if all(saved[c] >= per_class for c in class_names):
            break

    if any(saved[name] != per_class for name in class_names):
        raise RuntimeError(
            f"Could not prepare {per_class} images per class for {output_name}. "
            f"Saved={dict(saved)}, scanned={scanned}, skipped={dict(skipped)}"
        )

    summary = {
        "source_config": config_name,
        "source_split": source_split,
        "output_split": output_name,
        "counts": {name: int(saved[name]) for name in class_names},
        "rows_scanned": scanned,
        "skipped": dict(skipped),
        "source_model_rows_seen": {name: int(sum(generator_counts[name].values())) for name in class_names},
        "unique_source_model_names_seen": {name: len(generator_counts[name]) for name in class_names},
        "top_source_model_names_seen": {
            name: [{"model": model, "rows": int(count)} for model, count in generator_counts[name].most_common(20)]
            for name in class_names
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    global ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/content/deepguard-openfake"))
    parser.add_argument("--train-per-class", type=int, default=4000)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--ood-test-per-class", type=int, default=800)
    parser.add_argument("--wild-test-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--buffer-size", type=int, default=20000)
    ARGS = parser.parse_args()

    if min(
        ARGS.train_per_class,
        ARGS.val_per_class,
        ARGS.ood_test_per_class,
        ARGS.wild_test_per_class,
        ARGS.buffer_size,
    ) < 1:
        parser.error("sample counts and buffer size must be positive")

    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run pip install datasets pillow.") from exc

    ARGS.output.mkdir(parents=True, exist_ok=True)
    target_spec = [
        ("core", "train", "train", ARGS.train_per_class, ARGS.seed),
        ("core", "validation", "val", ARGS.val_per_class, ARGS.seed + 1),
        ("core", "test", "test_ood", ARGS.ood_test_per_class, ARGS.seed + 2),
        ("reddit", "test", "test_wild", ARGS.wild_test_per_class, ARGS.seed + 3),
    ]
    summaries = {}
    for config_name, source_split, output_name, count, seed in target_spec:
        summaries[output_name] = prepare_split(
            config_name, source_split, output_name, count, seed, ARGS.buffer_size
        )

    manifest = {
        "dataset": DATASET_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "train": "OpenFake core/train only",
            "val": "OpenFake core/validation; used for checkpoint/threshold selection",
            "test_ood": "OpenFake core/test; held-out generators and different real-image sources",
            "test_wild": "OpenFake reddit/test; separate in-the-wild test, never used for training or threshold selection",
        },
        "sample_counts": {split: item["counts"] for split, item in summaries.items()},
        "seed": ARGS.seed,
        "buffer_size": ARGS.buffer_size,
        "imagefolder_class_order": ["ai_generated", "real"],
        "split_summaries": summaries,
        "license_and_use_notes": [
            "Dataset card top-level metadata reports CC-BY-NC-4.0; card also describes source/generator-specific restrictions. Review current terms.",
            "Academic/non-commercial experiment only unless rights for the relevant records and generators are separately cleared.",
            "Do not commit or redistribute sampled raw images. Do not publish the trained checkpoint until data and model terms have been reviewed.",
            "OpenFake core/test is reserved for out-of-distribution evaluation; do not train on it, tune thresholds on it, or select epochs by its score.",
            "The reddit/test config is a separate in-the-wild evaluation set and may contain label noise from subreddit-context labeling.",
            "This experiment does not replace the existing Tiny-GenImage baseline; report results separately.",
        ],
    }
    manifest_path = ARGS.output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}")
    print(json.dumps(manifest["sample_counts"], indent=2))


if __name__ == "__main__":
    main()
