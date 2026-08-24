"""Inference for the DEEP-Guard image classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


CLASSES = ["ai_generated", "real"]
DROPOUT = 0.2
# Do not expose raw softmax as if it were a calibrated probability.
# This threshold intentionally creates an uncertainty band to reduce
# overconfident false positives on images outside the training distribution.
AI_THRESHOLD = 0.75
REAL_THRESHOLD = 0.35


def _build_model(num_classes: int) -> torch.nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(DROPOUT),
        torch.nn.Linear(in_features, num_classes),
    )
    return model


def _verdict(ai_score: float) -> tuple[str, str]:
    """Map the raw AI score to a conservative user-facing verdict.

    The model score is not a probability of provenance. An uncertainty band
    is preferable to forcing borderline predictions into a binary decision.
    """
    if ai_score >= AI_THRESHOLD:
        return "likely_ai_generated", "strong_ai_signal"
    if ai_score <= REAL_THRESHOLD:
        return "likely_real", "strong_real_signal"
    return "uncertain", "mixed_model_signal"


def predict(path: Path, checkpoint_path: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model_state_dict"]
    classes = checkpoint.get("classes", CLASSES)

    model = _build_model(len(classes))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    weights = EfficientNet_B0_Weights.DEFAULT
    image = Image.open(path).convert("RGB")
    tensor = weights.transforms()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    ai_index = classes.index("ai_generated")
    real_index = classes.index("real")
    ai_score = float(probabilities[ai_index].item())
    real_score = float(probabilities[real_index].item())
    predicted_index = int(torch.argmax(probabilities).item())
    raw_label = classes[predicted_index]
    verdict, signal = _verdict(ai_score)

    return {
        "label": raw_label,
        "verdict": verdict,
        "confidence": max(ai_score, real_score),
        "ai_score": ai_score,
        "real_score": real_score,
        "signal": signal,
        "model": "deepguard-image-v1",
        "classes": classes,
        "calibrated": False,
        "note": "Model score is not proof of image provenance; uncertain is used for borderline predictions.",
    }
