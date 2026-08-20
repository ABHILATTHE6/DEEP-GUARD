"""Image dataset utilities for folder-based real/AI training data."""

from pathlib import Path

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .config import ImageModelConfig


def build_transforms(config: ImageModelConfig, train: bool) -> transforms.Compose:
    """Build deterministic validation or augmented training transforms."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_image_dataset(root: str | Path, config: ImageModelConfig, train: bool) -> Dataset:
    """Load ImageFolder data with classes ``real`` and ``ai_generated``."""
    return datasets.ImageFolder(root=str(root), transform=build_transforms(config, train))
