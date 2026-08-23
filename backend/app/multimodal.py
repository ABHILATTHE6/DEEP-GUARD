"""Application-level orchestration for DEEP-Guard modalities."""
from __future__ import annotations
from typing import Any
from ml.fusion.engine import fuse


def aggregate(image: dict[str, Any] | None = None, video: dict[str, Any] | None = None, audio: dict[str, Any] | None = None) -> dict[str, Any]:
    results = {k: v for k, v in {"image": image, "video": video, "audio": audio}.items() if v}
    return {"modalities": results, "fusion": fuse(results)}
