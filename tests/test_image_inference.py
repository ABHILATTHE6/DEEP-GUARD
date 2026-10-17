import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")

from PIL import Image
import torch

from ml.image.config import ImageModelConfig
from ml.image.dataset import build_transforms
from ml.image.inference import preprocess_image


def test_inference_preprocessing_matches_validation_pipeline() -> None:
    image = Image.new("RGB", (389, 271), color=(40, 120, 200))

    actual = preprocess_image(image)
    expected = build_transforms(ImageModelConfig(pretrained=False), train=False)(image)

    assert actual.shape == (3, 224, 224)
    assert torch.equal(actual, expected)
