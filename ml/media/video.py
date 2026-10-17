"""Experimental video screening by sampling frames with the image baseline.

This is a frame-aggregated baseline, not a temporal video model. Its output must
not be described as a validated detector for every generated-video family.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image

from ml.image.inference import predict_images


def analyze_video(path: Path, image_checkpoint: Path, sample_count: int = 8) -> dict[str, Any]:
    """Sample up to sample_count frames and aggregate the existing image model scores."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video support requires OpenCV. Install the project ML dependencies and restart the API."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("The uploaded video could not be opened. Try an MP4 or WebM file.")

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames < 1:
            raise ValueError("The uploaded video has no readable frames.")

        frame_indices = sorted({
            int(round(i * max(total_frames - 1, 0) / max(sample_count - 1, 1)))
            for i in range(min(sample_count, total_frames))
        })
        frames: list[Image.Image] = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
    finally:
        cap.release()

    if not frames:
        raise ValueError("No readable frames were found in the uploaded video.")

    predictions = predict_images(frames, image_checkpoint)
    ai_scores = [float(item["ai_probability"]) for item in predictions]
    real_scores = [float(item["real_probability"]) for item in predictions]
    ai_probability = mean(ai_scores)
    real_probability = mean(real_scores)

    # This 0.5 decision boundary is a simple prototype rule; image validation
    # calibration is not valid calibration evidence for full video clips.
    if abs(ai_probability - 0.5) <= 0.05:
        verdict = "uncertain"
    elif ai_probability > 0.5:
        verdict = "likely_ai_generated"
    else:
        verdict = "likely_real"

    duration = total_frames / fps if fps > 0 else None
    confidence = max(ai_probability, real_probability)
    evidence = [
        f"Scored {len(predictions)} evenly sampled frames with the DEEP-GUARD image baseline.",
        f"Mean AI-generated frame score: {ai_probability * 100:.1f}%.",
        f"Mean real-class frame score: {real_probability * 100:.1f}%.",
        f"Variation across sampled frame AI scores (population standard deviation): {pstdev(ai_scores):.3f}.",
        "Experimental frame aggregation only: temporal inconsistencies and audio/video synchronization were not analyzed.",
        "The image model's validation threshold has not been calibrated for videos; treat the video verdict as a screening signal.",
    ]
    if duration is not None:
        evidence.insert(0, f"Video duration: {duration:.1f} seconds.")

    if verdict == "uncertain":
        summary = (
            "The sampled-frame scores are close to the prototype decision boundary. "
            "DEEP-GUARD cannot make a confident video-level determination from this clip."
        )
    elif verdict == "likely_ai_generated":
        summary = (
            f"The sampled frames produced a mean AI-generated score of {ai_probability * 100:.1f}%. "
            "This is an experimental frame-based screening result, not a validated temporal video verdict."
        )
    else:
        summary = (
            f"The sampled frames produced a mean real-class score of {real_probability * 100:.1f}%. "
            "This does not establish that the full video is authentic."
        )

    return {
        "modality": "video",
        "verdict": verdict,
        "confidence": confidence,
        "status": "experimental_frame_based",
        "model": "deepguard-image-efficientnet-b0-frame-aggregate",
        "evidence": evidence,
        "summary": summary,
        "scores": {"ai_generated": ai_probability, "real": real_probability},
        "frames_sampled": len(predictions),
        "duration_seconds": round(duration, 2) if duration is not None else None,
        "explainability": {
            "available": False,
            "message": "Per-frame temporal heatmaps are not available in this baseline.",
        },
    }
