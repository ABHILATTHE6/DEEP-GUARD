"""Modality routing and evidence contract for DEEP-Guard."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from ml.image.inference import predict

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
            "summary": "DEEP-Guard could not assess this image because the trained image checkpoint is not configured.",
        }

    result = predict(path, checkpoint)
    label = str(result["label"])
    confidence = float(result["confidence"])
    ai_probability = float(result.get("ai_probability", confidence if label == "ai_generated" else 1.0 - confidence))
    real_probability = 1.0 - ai_probability
    verdict = "likely_ai_generated" if label == "ai_generated" else "likely_real"

    if verdict == "likely_ai_generated":
        summary = (
            f"The image is classified as likely AI-generated with {confidence * 100:.1f}% model confidence. "
            f"The model assigns approximately {ai_probability * 100:.1f}% probability to the AI-generated class "
            f"and {real_probability * 100:.1f}% to the real class."
        )
    else:
        summary = (
            f"The image is classified as likely real with {confidence * 100:.1f}% model confidence. "
            f"The model assigns approximately {real_probability * 100:.1f}% probability to the real class "
            f"and {ai_probability * 100:.1f}% to the AI-generated class."
        )

    evidence = [
        "EfficientNet-B0 transfer-learning model prediction.",
        f"AI-generated class score: {ai_probability * 100:.1f}%.",
        f"Real class score: {real_probability * 100:.1f}%.",
        "The score represents model confidence, not definitive proof of image origin.",
    ]

    return {
        "modality": "image",
        "verdict": verdict,
        "confidence": confidence,
        "status": "ready",
        "model": "deepguard-image-v1",
        "evidence": evidence,
        "summary": summary,
        "scores": {
            "ai_generated": ai_probability,
            "real": real_probability,
        },
        "explainability": {
            "available": False,
            "message": "Pixel-level heatmaps are not enabled in the current baseline model.",
        },
    }


def _untrained(modality: str, model: str) -> dict:
    return {
        "modality": modality,
        "verdict": "uncertain",
        "confidence": None,
        "status": "experimental_model_not_trained",
        "model": model,
        "evidence": [f"{modality.title()} model is not yet trained; no synthetic result is claimed."],
        "summary": f"The {modality} pipeline is present, but its model has not yet been trained, so DEEP-Guard is not making a detection claim.",
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
