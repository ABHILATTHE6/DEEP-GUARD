"""Transfer-learning model used by the first DEEP-Guard image baseline."""

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import ImageModelConfig


def build_model(config: ImageModelConfig) -> nn.Module:
    """Create an EfficientNet-B0 classifier for real vs AI-generated images."""
    weights = EfficientNet_B0_Weights.DEFAULT if config.pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(config.dropout),
        nn.Linear(in_features, config.num_classes),
    )
    return model
