# DEEP-Guard Examiner Demo

## Working capabilities

1. Image training pipeline: EfficientNet-B0, reproducible evaluation and checkpointing.
2. Image API/dashboard: upload, assessment, confidence, model/status metadata.
3. Video pipeline: frame sampling and image-model score aggregation when a trained image checkpoint is configured.
4. Audio pipeline: waveform loading, Mel-spectrogram feature extraction and explicit experimental state until a trained audio model is supplied.
5. Multimodal fusion: weighted evidence aggregation with an `UNCERTAIN` outcome for insufficient evidence or high disagreement.
6. Grad-CAM utility for model-influential regions.

## Run locally

### Backend

```bash
uvicorn backend.app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set `DEEP_GUARD_IMAGE_CHECKPOINT` to a real checkpoint produced by the training pipeline to activate image inference. Do not claim benchmark performance until a real dataset training/evaluation run has produced metrics.

## Examiner message

DEEP-Guard is not a general-purpose chatbot. It is a specialized multimodal authenticity-assessment pipeline combining task-specific deep-learning models, explainability, uncertainty-aware fusion, robustness evaluation and media provenance signals. It reports an assessment rather than absolute proof.
