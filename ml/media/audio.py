"""Audio preprocessing and explicit-state synthetic speech assessment."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def analyze_audio(path: Path, model: Any | None = None) -> dict[str, Any]:
    try:
        import librosa
        import numpy as np
        y, sr = librosa.load(path, sr=16000, mono=True)
        duration = float(len(y) / sr) if sr else 0.0
        rms = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y))) if len(y) else 0.0
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        features = {"duration_seconds": round(duration, 2), "sample_rate": sr, "rms": round(rms, 6), "zero_crossing_rate": round(zcr, 6), "mel_shape": list(mel.shape)}
        if model is None:
            return {"status": "experimental", "verdict": "UNCERTAIN", "reason": "Audio model is not trained", "features": features}
        probability = float(model.predict(mel))
        return {"status": "active", "verdict": "LIKELY SYNTHETIC" if probability >= 0.5 else "LIKELY REAL", "ai_probability": round(probability, 6), "features": features}
    except Exception as exc:
        return {"status": "experimental", "verdict": "UNCERTAIN", "reason": str(exc)}
