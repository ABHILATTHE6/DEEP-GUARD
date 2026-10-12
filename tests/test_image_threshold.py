import numpy as np
import pytest

from ml.image.metrics import calibrate_real_probability_threshold


def test_threshold_can_separate_perfectly_scored_classes() -> None:
    labels = np.array([0, 0, 1, 1])  # ai_generated=0, real=1
    probabilities = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
    ])

    threshold, ai_recall, real_fpr = calibrate_real_probability_threshold(
        labels,
        probabilities,
        ["ai_generated", "real"],
        target_real_fpr=0.0,
    )

    assert threshold == pytest.approx(0.5)
    assert ai_recall == pytest.approx(1.0)
    assert real_fpr == pytest.approx(0.0)


def test_threshold_respects_real_false_positive_cap() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([
        [0.9, 0.1],
        [0.7, 0.3],
        [0.4, 0.6],
        [0.6, 0.4],
        [0.3, 0.7],
        [0.1, 0.9],
    ])

    threshold, ai_recall, real_fpr = calibrate_real_probability_threshold(
        labels,
        probabilities,
        ["ai_generated", "real"],
        target_real_fpr=1 / 3,
    )

    assert threshold == pytest.approx(0.65)
    assert ai_recall == pytest.approx(1.0)
    assert real_fpr <= 1 / 3


def test_threshold_rejects_validation_data_missing_a_class() -> None:
    with pytest.raises(ValueError, match="both AI-generated and real"):
        calibrate_real_probability_threshold(
            np.array([0, 0]),
            np.array([[0.9, 0.1], [0.8, 0.2]]),
            ["ai_generated", "real"],
            target_real_fpr=0.03,
        )


def test_threshold_rejects_invalid_fpr_target() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_real_probability_threshold(
            np.array([0, 1]),
            np.array([[0.9, 0.1], [0.1, 0.9]]),
            ["ai_generated", "real"],
            target_real_fpr=2.0,
        )
