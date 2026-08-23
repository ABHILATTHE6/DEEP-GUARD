"""Lightweight video analysis built from the validated image detector.

This module samples frames and aggregates image-model probabilities. It is deliberately
model-agnostic and returns an explicit untrained state when no image checkpoint exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def analyze_video(path: Path, image_predictor: Any | None = None, sample_count: int = 12) -> dict[str, Any]:
    try:
        import cv2
    except ImportError:
        return {"status": "experimental", "verdict": "UNCERTAIN", "reason": "OpenCV is not installed"}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "error", "verdict": "UNCERTAIN", "reason": "Unable to open video"}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = total / fps if fps else 0.0
    indices = set(int(i * max(total - 1, 0) / max(sample_count - 1, 1)) for i in range(min(sample_count, max(total, 1))))
    scores: list[float] = []
    analyzed = 0
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no in indices and image_predictor is not None:
            try:
                result = image_predictor(frame)
                if result.get("ai_probability") is not None:
                    scores.append(float(result["ai_probability"]))
                analyzed += 1
            except Exception:
                pass
        frame_no += 1
    cap.release()
    if not scores:
        return {"status": "experimental", "verdict": "UNCERTAIN", "reason": "No trained image model available for frame scoring", "frames_sampled": len(indices), "duration_seconds": round(duration, 2)}
    ai = sum(scores) / len(scores)
    verdict = "LIKELY AI-GENERATED" if ai >= 0.5 else "LIKELY REAL"
    return {"status": "active", "verdict": verdict, "ai_probability": round(ai, 6), "frames_sampled": len(indices), "frames_scored": analyzed, "duration_seconds": round(duration, 2), "frame_scores": [round(s, 4) for s in scores]}
