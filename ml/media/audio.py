"""Audio deepfake screening using a pretrained speech anti-spoofing model.

The downloaded upstream weights are CC-BY-NC-SA 4.0. They are fetched lazily
from Hugging Face and are not included in this repository or Docker image.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_ID = "SpeechAntiSpoofingBenchmarks/Wav2Vec2-Small-AntiDeepfake"
MODEL_REVISION = "85f8e14"  # pinned Hugging Face model revision
SAMPLE_RATE = 16000
MAX_ANALYZED_SECONDS = 30.0
MIN_ANALYZED_SECONDS = 1.0


@lru_cache(maxsize=1)
def _load_detector() -> tuple[Any, Any, Any]:
    """Load the audio processor and classifier only on the first audio request."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    try:
        processor = AutoFeatureExtractor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
    except (OSError, ValueError):
        # The upstream model is a Wav2Vec2 checkpoint; the base feature extractor
        # provides its standard 16 kHz waveform normalization if no processor file exists.
        from transformers import Wav2Vec2FeatureExtractor

        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

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
    """Resolve fake/real indices from model labels; fallback follows upstream model card."""
    id2label = getattr(model.config, "id2label", {}) or {}
    names = {int(index): str(label).lower() for index, label in id2label.items()}
    fake_terms = ("fake", "spoof", "synthetic", "deepfake", "generated")
    real_terms = ("real", "bonafide", "bona fide", "genuine", "human")
    fake = next((i for i, name in names.items() if any(term in name for term in fake_terms)), None)
    real = next((i for i, name in names.items() if any(term in name for term in real_terms)), None)
    if fake is not None and real is not None and fake != real:
        return fake, real
    # The NII AntiDeepfake model card defines its binary scores as class 0=fake, 1=real.
    return 0, 1


def analyze_audio(path: Path, model: Any | None = None) -> dict[str, Any]:
    """Analyze a short speech clip; output remains experimental pending local evaluation."""
    try:
        import librosa
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Audio support requires librosa, soundfile, transformers and PyTorch. "
            "Install the project ML dependencies and restart the API."
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
            "evidence": ["Audio clips shorter than one second are not assessed."],
            "summary": "The audio is too short for a meaningful speech deepfake assessment.",
            "explainability": {"available": False, "message": "Audio attribution plots are not available."},
        }

    if model is None:
        try:
            processor, detector, device = _load_detector()
        except Exception as exc:
            # Do not silently substitute hand-crafted audio statistics for a detector.
            return {
                "modality": "audio",
                "verdict": "uncertain",
                "confidence": None,
                "status": "model_unavailable",
                "model": MODEL_ID,
                "evidence": [
                    "The pretrained audio model could not be loaded.",
                    "An internet connection is needed on first use to download the pinned model weights.",
                    "No authenticity verdict was fabricated.",
                ],
                "summary": (
                    "DEEP-GUARD could not run the pretrained audio detector. "
                    "Check internet access and the Python dependency installation, then retry."
                ),
                "explainability": {"available": False, "message": "Audio attribution plots are not available."},
            }
    else:
        processor = None
        detector = model
        device = next(detector.parameters()).device

    waveform = np.asarray(waveform, dtype=np.float32)
    duration = len(waveform) / SAMPLE_RATE
    if duration > MAX_ANALYZED_SECONDS:
        waveform = waveform[: int(MAX_ANALYZED_SECONDS * SAMPLE_RATE)]
        duration = MAX_ANALYZED_SECONDS

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
        f"Pretrained Wav2Vec2 anti-spoofing model: {MODEL_ID}.",
        f"AI-generated / spoof speech score: {fake_probability * 100:.1f}%.",
        f"Human / bona-fide speech score: {real_probability * 100:.1f}%.",
        f"Analyzed {duration:.1f} seconds at {SAMPLE_RATE} Hz mono; longer files are truncated to the first {MAX_ANALYZED_SECONDS:.0f} seconds.",
        "This external model is licensed CC-BY-NC-SA 4.0 and is intended for non-commercial research/education.",
        "No DEEP-GUARD-specific audio benchmark has been measured yet; results may not generalize to unseen voices, languages, codecs, or cloning tools.",
    ]
    summary = (
        "The pretrained speech model is close to its default 0.5 decision boundary; treat this clip as inconclusive."
        if verdict == "uncertain"
        else (
            f"The pretrained speech model assigned {fake_probability * 100:.1f}% to its synthetic/spoof class. "
            "This is a screening signal, not proof that the voice was generated."
            if verdict == "likely_ai_generated"
            else f"The pretrained speech model assigned {real_probability * 100:.1f}% to its human/bona-fide class. "
            "This is not proof of a genuine speaker or recording."
        )
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
