"""Modality routing and evidence contract for DEEP-Guard."""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from ml.image.inference import predict

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-matroska"}
AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/webm"}


class UnsupportedMediaTypeError(ValueError):
    """Raised when an uploaded file is not a supported image, video, or audio type."""


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
            "summary": (
                "DEEP-Guard could not assess this image because the trained image checkpoint "
                "is not configured."
            ),
        }

    result = predict(Path(path), Path(checkpoint))
    label = str(result.get("label", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    ai_probability = float(
        result.get(
            "ai_probability",
            result.get("ai_score", confidence if label == "ai_generated" else 1.0 - confidence),
        )
    )
    real_probability = float(
        result.get("real_probability", result.get("real_score", 1.0 - ai_probability))
    )
    verdict = str(result.get("verdict", "uncertain"))
    model = str(result.get("model", "deepguard-image-unknown"))
    threshold = result.get("decision_threshold")

    if verdict == "likely_ai_generated":
        summary = (
            f"The image is classified as likely AI-generated with {confidence * 100:.1f}% "
            f"maximum class probability. The model assigns {ai_probability * 100:.1f}% to "
            f"the AI-generated class and {real_probability * 100:.1f}% to the real class."
        )
    elif verdict == "likely_real":
        summary = (
            f"The image is classified as likely real with {confidence * 100:.1f}% maximum "
            f"class probability. The model assigns {real_probability * 100:.1f}% to the real "
            f"class and {ai_probability * 100:.1f}% to the AI-generated class."
        )
    else:
        summary = (
            "The image assessment is uncertain. The model's scores are close to its decision "
            "boundary or do not support a confident binary verdict. Review the original source "
            "and additional context."
        )

    evidence = [
        "EfficientNet-B0 transfer-learning model prediction.",
        f"AI-generated class score: {ai_probability * 100:.1f}%.",
        f"Real class score: {real_probability * 100:.1f}%.",
        "Model scores are not definitive proof of image origin.",
    ]
    if threshold is not None:
        evidence.append(
            f"Validation-calibrated decision threshold on P(real): {float(threshold):.4f}."
        )

    return {
        "modality": "image",
        "verdict": verdict,
        "confidence": confidence,
        "status": "ready",
        "model": model,
        "evidence": evidence,
        "summary": summary,
        "scores": {
            "ai_generated": ai_probability,
            "real": real_probability,
        },
        "decision_threshold": threshold,
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
        "summary": (
            f"The {modality} pipeline is present, but its model has not yet been trained, "
            "so DEEP-Guard is not making a detection claim."
        ),
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
    raise UnsupportedMediaTypeError(f"Unsupported media type: {content_type}")
