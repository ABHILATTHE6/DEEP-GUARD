"""Modality routing and evidence contract for DEEP-Guard."""
from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from PIL import UnidentifiedImageError

from ml.image.inference import predict
from ml.media.audio import analyze_audio
from ml.media.video import analyze_video

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_TYPES = {
    "video/mp4", "video/webm", "video/quicktime", "video/x-matroska",
    "video/x-msvideo", "video/avi",
}
AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/flac", "audio/mpeg", "audio/mp4",
    "audio/ogg", "audio/webm", "audio/x-m4a", "audio/m4a",
}


class UnsupportedMediaTypeError(ValueError):
    """Raised when an uploaded file is not a supported image, video, or audio type."""


class InvalidMediaFileError(ValueError):
    """Raised when an upload claims a supported media type but cannot be decoded."""


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
                "DEEP-GUARD could not assess this image because the trained image checkpoint "
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
        "scores": {"ai_generated": ai_probability, "real": real_probability},
        "decision_threshold": threshold,
        "explainability": {
            "available": False,
            "message": "Pixel-level heatmaps are not enabled in the current baseline model.",
        },
    }


def _video(path: str) -> dict:
    checkpoint = os.getenv("DEEP_GUARD_IMAGE_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        return {
            "modality": "video",
            "verdict": "uncertain",
            "confidence": None,
            "status": "model_not_configured",
            "model": "deepguard-video-frame-baseline",
            "evidence": ["No image checkpoint is configured for sampled-frame scoring."],
            "summary": "Video screening requires the current DEEP-GUARD image checkpoint.",
        }
    try:
        return analyze_video(Path(path), Path(checkpoint))
    except ValueError as exc:
        raise InvalidMediaFileError(str(exc)) from exc


def _audio(path: str) -> dict:
    try:
        return analyze_audio(Path(path))
    except ValueError as exc:
        raise InvalidMediaFileError(str(exc)) from exc


def _route_type(path: str, content_type: str) -> str:
    mime = (content_type or "").lower().split(";", maxsplit=1)[0].strip()
    if mime in IMAGE_TYPES or mime.startswith("image/"):
        return "image"
    if mime in VIDEO_TYPES or mime.startswith("video/"):
        return "video"
    if mime in AUDIO_TYPES or mime.startswith("audio/"):
        return "audio"

    guessed, _ = mimetypes.guess_type(path)
    guessed = guessed or ""
    if guessed.startswith("image/"):
        return "image"
    if guessed.startswith("video/"):
        return "video"
    if guessed.startswith("audio/"):
        return "audio"
    raise UnsupportedMediaTypeError(f"Unsupported media type: {content_type or guessed or 'unknown'}")


def analyze_file(path: str, content_type: str) -> dict:
    """Route a decoded file to image, video-frame, or speech inference."""
    media_type = _route_type(path, content_type)
    if media_type == "image":
        try:
            return _image(path)
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidMediaFileError("The uploaded image is not a valid, readable image file.") from exc
    if media_type == "video":
        return _video(path)
    return _audio(path)
