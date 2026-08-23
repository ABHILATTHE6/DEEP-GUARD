"""Modality routing and evidence contract for DEEP-Guard."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from ml.image.inference.predict import predict_image

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}
AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/webm"}


def _image(path: str) -> dict:
    checkpoint = os.getenv("DEEP_GUARD_IMAGE_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).exists():
        return {
            "modality": "image",
            "verdict": "uncertain",
            "confidence": None,
            "status": "model_not_configured",
            "model": "deepguard-image-v1",
            "evidence": ["No trained checkpoint configured for API inference."],
        }
    result = predict_image(path, checkpoint)
    return {
        "modality": "image",
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "status": "ready",
        "model": "deepguard-image-v1",
        "evidence": ["EfficientNet-B0 model prediction"],
    }


def _untrained(modality: str, model: str) -> dict:
    return {
        "modality": modality,
        "verdict": "uncertain",
        "confidence": None,
        "status": "experimental_model_not_trained",
        "model": model,
        "evidence": [f"{modality.title()} model is not yet trained; no synthetic result is claimed."],
    }


def analyze_file(path: str, content_type: str) -> dict:
    if content_type in IMAGE_TYPES or content_type.startswith("image/"):
        return _image(path)
    if content_type in VIDEO_TYPES or content_type.startswith("video/"):
        return _untrained("video", "deepguard-video-v1")
    if content_type in AUDIO_TYPES or content_type.startswith("audio/"):
        return _untrained("audio", "deepguard-audio-v1")
    guessed, _ = mimetypes.guess_type(path)
    if guessed and guessed.startswith("image/"):
        return _image(path)
    raise ValueError(f"Unsupported media type: {content_type}")
