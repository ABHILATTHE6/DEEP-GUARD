"""Tests for the image detector architecture."""

import pytest

pytest.importorskip("torch")

from ml.image.config import ImageModelConfig
from ml.image.model import build_model


def test_image_model_outputs_two_classes() -> None:
    import torch

    config = ImageModelConfig(pretrained=False)
    model = build_model(config)
    output = model(torch.randn(2, 3, config.image_size, config.image_size))
    assert output.shape == (2, 2)
