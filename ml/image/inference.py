"""Inference utilities for the DEEP-Guard image baseline."""

from pathlib import Path

import torch
from PIL import Image

from .config import ImageModelConfig
from .dataset import build_transforms
from .model import build_model


def predict(image_path: str | Path, checkpoint: str | Path, config: ImageModelConfig | None = None) -> dict[str, float | str]:
    """Return a class prediction and confidence for one image."""
    config = config or ImageModelConfig(pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    tensor = build_transforms(config, train=False)(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    index = int(probabilities.argmax().item())
    return {
        "label": config.class_names[index],
        "confidence": float(probabilities[index].item()),
    }
