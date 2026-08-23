"""Grad-CAM utility for torchvision classifiers."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def explain(model: Any, image: Any, target_layer: Any | None = None) -> dict[str, Any]:
    try:
        import torch
        if target_layer is None:
            target_layer = model.features[-1]
        activations = []
        gradients = []
        def fw(_, __, output): activations.append(output.detach())
        def bw(_, __, grad_output): gradients.append(grad_output[0].detach())
        h1 = target_layer.register_forward_hook(fw)
        h2 = target_layer.register_full_backward_hook(bw)
        logits = model(image)
        idx = int(logits.argmax(dim=1)[0])
        model.zero_grad(set_to_none=True)
        logits[:, idx].sum().backward()
        h1.remove(); h2.remove()
        a, g = activations[-1], gradients[-1]
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return {"class_index": idx, "heatmap": cam.tolist(), "status": "active"}
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
