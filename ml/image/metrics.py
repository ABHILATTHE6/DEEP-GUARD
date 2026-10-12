"""Threshold calibration helpers shared by training and evaluation."""

from __future__ import annotations

import numpy as np


def calibrate_real_probability_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    target_real_fpr: float,
) -> tuple[float, float, float]:
    """Maximize AI recall while keeping real images misclassified as AI under an FPR cap.

    Args:
        labels: Integer ImageFolder class indices.
        probabilities: NxC probability matrix in the supplied class order.
        classes: Class names corresponding to probability columns.
        target_real_fpr: Maximum allowed validation false-positive rate for real images.

    Returns:
        A tuple of (P(real) decision threshold, AI recall, real-image FPR).
        Predict real when P(real) >= threshold; otherwise predict AI-generated.
    """
    if not 0.0 <= target_real_fpr <= 1.0:
        raise ValueError("target_real_fpr must be between 0 and 1")
    if "ai_generated" not in classes or "real" not in classes:
        raise ValueError(f"classes must include 'ai_generated' and 'real'; got {classes}")
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities must be an NxC matrix with one row per label")
    if probabilities.shape[1] != len(classes):
        raise ValueError("probability column count must match classes")
    if labels.size == 0:
        raise ValueError("cannot calibrate a threshold on an empty validation split")

    ai_index = classes.index("ai_generated")
    real_index = classes.index("real")
    ai_mask = labels == ai_index
    real_mask = labels == real_index
    real_scores = probabilities[:, real_index]

    if not ai_mask.any() or not real_mask.any():
        raise ValueError("Validation split must contain both AI-generated and real images.")

    unique_scores = np.unique(real_scores)
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.unique(np.concatenate(([0.0], midpoints, [1.0])))

    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted_ai = real_scores < threshold
        ai_recall = float(predicted_ai[ai_mask].mean())
        real_fpr = float(predicted_ai[real_mask].mean())
        if real_fpr <= target_real_fpr:
            candidate = (ai_recall, -real_fpr, -float(threshold))
            if best is None or candidate > best:
                best = candidate

    # Threshold 0 always gives a feasible no-positive-predictions baseline for
    # finite softmax probabilities, so this is a defensive fallback.
    if best is None:
        return 0.0, 0.0, 0.0
    return -best[2], best[0], -best[1]
