"""Inference for the DEEP-Guard image classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision.models import efficientnet_b0

from ml.image.config import ImageModelConfig
from ml.image.dataset import build_transforms


CLASSES = ["ai_generated", "real"]
DROPOUT = 0.2
AI_THRESHOLD = 0.75
REAL_THRESHOLD = 0.35
UNCERTAINTY_MARGIN = 0.03


def _build_model(num_classes: int) -> torch.nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(in_features, num_classes),
    )
    return model


def _verdict(
    ai_score: float,
    real_score: float,
    calibrated_real_threshold: float | None = None,
) -> tuple[str, str]:
    """Use validation-calibrated threshold when present; otherwise use fallback bands."""
    if calibrated_real_threshold is not None:
        if abs(real_score - calibrated_real_threshold) <= UNCERTAINTY_MARGIN:
            return "uncertain", "near_calibrated_threshold"
        if real_score >= calibrated_real_threshold:
            return "likely_real", "validation_calibrated_real_threshold"
        return "likely_ai_generated", "validation_calibrated_ai_threshold"

    if ai_score >= AI_THRESHOLD:
        return "likely_ai_generated", "fallback_strong_ai_signal"
    if ai_score <= REAL_THRESHOLD:
        return "likely_real", "fallback_strong_real_signal"
    return "uncertain", "fallback_mixed_model_signal"


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the same deterministic preprocessing used for validation and test data."""
    config = ImageModelConfig(pretrained=False)
    transform = build_transforms(config, train=False)
    return transform(image)


def predict_images(images: Sequence[Image.Image], checkpoint_path: Path) -> list[dict]:
    """Batch-infer one or more PIL images while loading the checkpoint only once.

    This is used by the video frame sampler to avoid reloading the 46 MB checkpoint
    separately for every frame.
    """
    if not images:
        return []
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Image checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    classes = list(checkpoint.get("classes", CLASSES))
    if set(classes) != {"ai_generated", "real"}:
        raise ValueError(f"Unsupported checkpoint class mapping: {classes}")

    model = _build_model(len(classes))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tensors = torch.stack([preprocess_image(image.convert("RGB")) for image in images]).to(device)
    with torch.inference_mode():
        logits = model(tensors)
        probabilities = torch.softmax(logits, dim=1).cpu()

    ai_index = classes.index("ai_generated")
    real_index = classes.index("real")
    threshold = checkpoint.get("real_probability_threshold")
    if threshold is not None:
        threshold = float(threshold)

    results: list[dict] = []
    for row in probabilities:
        ai_score = float(row[ai_index].item())
        real_score = float(row[real_index].item())
        predicted_index = int(torch.argmax(row).item())
        verdict, signal = _verdict(ai_score, real_score, threshold)
        results.append({
            "label": classes[predicted_index],
            "verdict": verdict,
            "confidence": max(ai_score, real_score),
            "ai_probability": ai_score,
            "real_probability": real_score,
            "ai_score": ai_score,
            "real_score": real_score,
            "decision_threshold": threshold,
            "signal": signal,
            "model": checkpoint.get("model_name", "deepguard-image-efficientnet-b0"),
            "classes": classes,
            "calibrated": threshold is not None,
            "note": (
                "Threshold was selected using validation data only; model scores are not proof "
                "of image provenance. Borderline scores are reported as uncertain."
            ),
        })
    return results


def predict(path: Path, checkpoint_path: Path) -> dict:
    """Infer a single image, preserving the existing image API contract."""
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    return predict_images([image], checkpoint_path)[0]
