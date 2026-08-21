"""Validation helpers for DEEP-Guard dataset metadata.

This module validates metadata only; it never downloads or redistributes datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SUPPORTED_MODALITIES = {"image", "video", "audio"}
SUPPORTED_LABELS = {"real", "manipulated", "ai_generated"}
REQUIRED_DATASET_FIELDS = {
    "id",
    "name",
    "modality",
    "task",
    "status",
    "source",
    "access",
    "license_note",
    "labels",
}


def load_registry(path: str | Path) -> dict[str, Any]:
    """Load a YAML dataset registry."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Registry root must be a mapping")
    return data


def validate_registry(registry: dict[str, Any]) -> list[str]:
    """Return validation errors; an empty list means the registry is valid."""
    errors: list[str] = []
    if registry.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")

    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return errors + ["datasets must be a non-empty list"]

    seen_ids: set[str] = set()
    for index, dataset in enumerate(datasets):
        prefix = f"datasets[{index}]"
        if not isinstance(dataset, dict):
            errors.append(f"{prefix} must be a mapping")
            continue

        missing = REQUIRED_DATASET_FIELDS - dataset.keys()
        errors.extend(f"{prefix} missing '{field}'" for field in sorted(missing))

        dataset_id = dataset.get("id")
        if isinstance(dataset_id, str):
            if dataset_id in seen_ids:
                errors.append(f"duplicate dataset id: {dataset_id}")
            seen_ids.add(dataset_id)

        modality = dataset.get("modality")
        if modality not in SUPPORTED_MODALITIES:
            errors.append(f"{prefix}.modality must be one of {sorted(SUPPORTED_MODALITIES)}")

        labels = dataset.get("labels")
        if not isinstance(labels, list) or not labels:
            errors.append(f"{prefix}.labels must be a non-empty list")
        elif any(label not in SUPPORTED_LABELS for label in labels):
            errors.append(f"{prefix}.labels contains unsupported labels")

    return errors


def validate_file(path: str | Path) -> None:
    """Raise ValueError when a registry file is invalid."""
    errors = validate_registry(load_registry(path))
    if errors:
        raise ValueError("Invalid dataset registry:\n- " + "\n- ".join(errors))
