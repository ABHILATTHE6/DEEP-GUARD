# DEEP-Guard

## Multimodal AI-Generated Media & Deepfake Detection

DEEP-Guard is a research and engineering platform for assessing the authenticity of digital media across **images, videos, and audio** using task-specific deep-learning pipelines.

> **Important:** DEEP-Guard provides an authenticity assessment, not an absolute guarantee. Detectors can fail on unseen generators, transformations, compression, adversarial manipulation, and distribution shifts.

## Current implementation status

| Capability | Status |
|---|---|
| Repository + project structure | ✅ Implemented |
| Image EfficientNet-B0 training/evaluation | ✅ Implemented |
| Image inference API | ✅ Implemented |
| React examiner dashboard | ✅ Implemented |
| Video frame sampling + image-model aggregation | ✅ Implemented |
| Audio loading + Mel-spectrogram extraction | ✅ Implemented |
| Multimodal evidence fusion | ✅ Implemented |
| Grad-CAM utility | ✅ Implemented |
| CI workflow | ✅ Implemented |
| Trained video checkpoint | ⏳ Requires dataset + training run |
| Trained audio checkpoint | ⏳ Requires dataset + training run |
| Cross-generator benchmark | ⏳ Requires evaluation data |
| Robustness benchmark | ⏳ Requires evaluation data |

The repository intentionally returns **UNCERTAIN/experimental** when a trained modality checkpoint is unavailable. It does not fabricate benchmark accuracy.

## Architecture

```text
                    DEEP-GUARD DASHBOARD
                              |
                           FastAPI
                              |
             +----------------+----------------+
             |                |                |
           IMAGE            VIDEO            AUDIO
             |                |                |
        EfficientNet      sampled frames   Mel spectrogram
        + Grad-CAM        + image model    + audio model
             |                |                |
             +----------------+----------------+
                              |
                       Evidence Fusion
                              |
                +-------------+-------------+
                |             |             |
             REAL       AI-GENERATED     UNCERTAIN
```

## How to complete the remaining experimental work

### 1. Prepare licensed datasets

Keep raw datasets outside Git. Register each dataset with its source, license, class definitions, generator/source information, and version. Create independent train/validation/test splits and prevent near-duplicate leakage.

### 2. Train the image baseline

Use the existing training pipeline and run a real experiment. Save the checkpoint and record accuracy, precision, recall, F1, ROC-AUC, confusion matrix, seed, dataset version, and configuration.

### 3. Activate video inference

Set `DEEP_GUARD_IMAGE_CHECKPOINT` to the trained image checkpoint. Run the video frame pipeline and evaluate it on real and manipulated videos. The current implementation samples frames and aggregates image-model probabilities; a temporal model can be added as the research upgrade.

### 4. Train the audio model

Use real and synthetic/cloned speech data with compatible licenses. Train a CNN/Transformer over Mel-spectrograms, save the checkpoint, then connect it to `ml/media/audio.py`.

### 5. Validate fusion

Evaluate image-only, video-only, audio-only, and multimodal performance. Calibrate thresholds on validation data and report disagreement/uncertainty instead of forcing a binary decision.

### 6. Robustness and generalization

Test JPEG compression, resizing, blur, noise, screenshots, transcoding, and at least one generator/domain not represented in training. Report the performance drop rather than only the best-case score.

### 7. Final examiner demonstration

Upload an image, video, and audio sample in the dashboard. Show the prediction, confidence, model version, evidence, Grad-CAM where applicable, uncertainty, and final report. Use measured results only.

## Local run

Backend:

```bash
uvicorn backend.app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Set `DEEP_GUARD_IMAGE_CHECKPOINT` to a real checkpoint produced by the training pipeline to activate image inference.

## Evaluation

Models should be evaluated with task-appropriate metrics including accuracy, precision, recall, F1, ROC-AUC, PR-AUC where appropriate, confusion matrices, calibration/error analysis, and unseen-generator or cross-dataset testing where feasible.

## Why DEEP-Guard is different

DEEP-Guard is not a general-purpose chatbot. It is a specialized authenticity-assessment system that combines modality-specific deep-learning models, explainability, uncertainty-aware evidence fusion, robustness testing, and media provenance signals. The project is designed to expose evidence and limitations rather than present a single unexplained binary answer.

## Responsible AI

A model prediction should not be treated as definitive proof of authenticity or fraud. Human review and source/context verification remain important for high-impact decisions.

## Author

**Abhishek Latthe** — [@ABHILATTHE6](https://github.com/ABHILATTHE6)
