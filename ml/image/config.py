"""Configuration for the DEEP-Guard image detector."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageModelConfig:
    image_size: int = 224
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.2
    class_names: tuple[str, str] = ("real", "ai_generated")
