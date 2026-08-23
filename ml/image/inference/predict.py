"""Single-image inference utilities for DEEP-Guard."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import models


def load_model(checkpoint_path: str | Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = [name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda item: item[1])]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), classes, device


def predict_image(image_path: str | Path, checkpoint_path: str | Path, threshold: float = 0.5) -> dict:
    model, classes, device = load_model(checkpoint_path)
    preprocess = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(image), dim=1)[0]
    index = int(probability.argmax())
    label = classes[index]
    confidence = float(probability[index])
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    if confidence < threshold:
        verdict = "uncertain"
    elif normalized in {"ai", "ai_generated", "fake", "synthetic"}:
        verdict = "likely_ai_generated"
    else:
        verdict = "likely_real"
    return {"verdict": verdict, "confidence": round(confidence, 6), "class": label, "model": "deepguard-image-efficientnet-b0"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(predict_image(args.image, args.checkpoint, args.threshold))


if __name__ == "__main__":
    main()
