"""Run single-image inference with a trained DEEP-Guard checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import models, transforms
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = [name for name, _ in sorted(checkpoint["class_to_idx"].items(), key=lambda item: item[1])]
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    preprocess = models.EfficientNet_B0_Weights.DEFAULT.transforms()
    image = preprocess(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probability = torch.softmax(model(image), dim=1)[0]

    index = int(probability.argmax())
    label = classes[index]
    confidence = float(probability[index])
    print({"verdict": label, "confidence": round(confidence, 6), "model": "deepguard-image-efficientnet-b0"})


if __name__ == "__main__":
    main()
