"""Uncertainty-aware multimodal score fusion."""
from __future__ import annotations
from typing import Any


def fuse(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scored = []
    weights = {"image": 0.45, "video": 0.35, "audio": 0.20}
    for modality, result in results.items():
        p = result.get("ai_probability")
        if isinstance(p, (int, float)):
            scored.append((modality, float(p), weights.get(modality, 1.0)))
    if not scored:
        return {"verdict": "UNCERTAIN", "status": "insufficient_evidence", "reason": "No trained modality produced a probability"}
    total_w = sum(w for _, _, w in scored)
    score = sum(p * w for _, p, w in scored) / total_w
    disagreement = max(p for _, p, _ in scored) - min(p for _, p, _ in scored) if len(scored) > 1 else 0.0
    if disagreement > 0.35 or 0.4 < score < 0.6:
        verdict = "UNCERTAIN"
    elif score >= 0.6:
        verdict = "LIKELY AI-GENERATED"
    else:
        verdict = "LIKELY REAL"
    return {"verdict": verdict, "ai_probability": round(score, 6), "disagreement": round(disagreement, 6), "modalities_used": [m for m, _, _ in scored], "status": "active"}
