"""Audio deepfake screening using an external pretrained Wav2Vec2 classifier.

The upstream checkpoint is Apache-2.0 licensed, but it is English-focused and
was fine-tuned on a small task-specific dataset. DEEP-GUARD has not independently
benchmarked this model. It downloads model weights on first use and is optional
for image/video-only runs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
MODEL_REVISION = "c5bffec702b5fc18b7b7523f1891036de198643a"
SAMPLE_RATE = 16000
MAX_ANALYZED_SECONDS = 13.0
MIN_ANALYZED_SECONDS = 2.5


@lru_cache(maxsize=1)
def _load_detector() -> tuple[Any, Any, Any]:
    """Load processor/model once per process; initial weights download is about 1.26 GB."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    processor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_safetensors=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device


def _class_indices(model: Any) -> tuple[int, int]:
    """Resolve fake/real indices from model labels; model card fallback is real=0, fake=1."""
    id2label = getattr(model.config, "id2label", {}) or {}
    names = {int(index): str(label).lower() for index, label in id2label.items()}
    fake_terms = ("fake", "spoof", "synthetic", "deepfake", "generated")
    real_terms = ("real", "bonafide", "bona fide", "genuine", "human")
    fake = next((i for i, name in names.items() if any(term in name for term in fake_terms)), None)
    real = next((i for i, name in names.items() if any(term in name for term in real_terms)), None)
    if fake is not None and real is not None and fake != real:
        return fake, real
    # Upstream model card documents output class 0 = real and class 1 = fake.
    return 1, 0


def analyze_audio(path: Path, model: Any | None = None) -> dict[str, Any]:
    """Analyze a speech clip; this is an external baseline, not a DEEP-GUARD-trained model."""
    try:
        import librosa
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Audio support requires librosa, soundfile, transformers and PyTorch. "
            "Install the project serving dependencies and restart the API."
        ) from exc

    try:
        waveform, sample_rate = librosa.load(
            str(path),
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_ANALYZED_SECONDS,
        )
    except Exception as exc:
        raise ValueError(
            "The uploaded audio could not be decoded. Try a WAV, FLAC, or MP3 speech file."
        ) from exc

    if waveform is None or len(waveform) < int(MIN_ANALYZED_SECONDS * SAMPLE_RATE):
        return {
            "modality": "audio",
            "verdict": "uncertain",
            "confidence": None,
            "status": "input_too_short",
            "model": MODEL_ID,
            "evidence": ["Audio clips shorter than 2.5 seconds are not assessed."],
            "summary": "The audio is too short for a meaningful speech deepfake assessment.",
            "explainability": {"available": False, "message": "Audio attribution plots are not available."},
        }

    try:
        if model is None:
            processor, detector, device = _load_detector()
        else:
            processor = None
            detector = model
            device = next(detector.parameters()).device

        waveform = np.asarray(waveform, dtype=np.float32)
        duration = min(len(waveform) / SAMPLE_RATE, MAX_ANALYZED_SECONDS)
        if len(waveform) > int(MAX_ANALYZED_SECONDS * SAMPLE_RATE):
            waveform = waveform[: int(MAX_ANALYZED_SECONDS * SAMPLE_RATE)]

        if processor is None:
            # Test-only model injection path.
            inputs = {"input_values": torch.from_numpy(waveform).unsqueeze(0).to(device)}
        else:
            inputs = processor(
                waveform,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = detector(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
    except Exception as exc:
        return {
            "modality": "audio",
            "verdict": "uncertain",
            "confidence": None,
            "status": "model_unavailable",
            "model": MODEL_ID,
            "evidence": [
                "The pretrained speech model could not be loaded or run.",
                "The first use requires an internet connection to download roughly 1.26 GB of weights.",
                "No authenticity verdict was fabricated.",
                f"Diagnostic: {type(exc).__name__}: {str(exc)[:240]}",
            ],
            "summary": (
                "DEEP-GUARD could not run the pretrained audio detector. Check internet access "
                "and Python dependencies, then retry. Image and video-frame analysis are independent."
            ),
            "explainability": {"available": False, "message": "Audio attribution plots are not available."},
        }

    fake_index, real_index = _class_indices(detector)
    if len(probabilities) < 2 or max(fake_index, real_index) >= len(probabilities):
        raise RuntimeError("The configured audio model did not return two class scores.")
    fake_probability = float(probabilities[fake_index])
    real_probability = float(probabilities[real_index])
    confidence = max(fake_probability, real_probability)
    verdict = "uncertain" if abs(fake_probability - 0.5) <= 0.05 else (
        "likely_ai_generated" if fake_probability > 0.5 else "likely_real"
    )

    evidence = [
        f"External pretrained Wav2Vec2 speech detector: {MODEL_ID}.",
        f"AI-generated / spoof speech score: {fake_probability * 100:.1f}%.",
        f"Human / bona-fide speech score: {real_probability * 100:.1f}%.",
        f"Analyzed {duration:.1f} seconds at {SAMPLE_RATE} Hz mono; longer files are truncated to the first {MAX_ANALYZED_SECONDS:.0f} seconds.",
        "The model card describes an English-focused model fine-tuned on 1,866 clips; small training/evaluation data limits generalization.",
        "DEEP-GUARD has not independently benchmarked this external model. Results may not generalize to unseen voices, languages, codecs, or cloning tools.",
    ]
    if verdict == "uncertain":
        summary = (
            "The pretrained speech model is close to its default 0.5 decision boundary; "
            "treat this clip as inconclusive."
        )
    elif verdict == "likely_ai_generated":
        summary = (
            f"The pretrained speech model assigned {fake_probability * 100:.1f}% to its synthetic/spoof class. "
            "This is a screening signal, not proof that the voice was generated."
        )
    else:
        summary = (
            f"The pretrained speech model assigned {real_probability * 100:.1f}% to its human/bona-fide class. "
            "This is not proof of a genuine speaker or recording."
        )

    return {
        "modality": "audio",
        "verdict": verdict,
        "confidence": confidence,
        "status": "experimental_pretrained",
        "model": MODEL_ID,
        "evidence": evidence,
        "summary": summary,
        "scores": {"ai_generated": fake_probability, "real": real_probability},
        "duration_seconds": round(duration, 2),
        "sample_rate": SAMPLE_RATE,
        "explainability": {"available": False, "message": "Pixel-level visual explanations do not apply to audio."},
    }
