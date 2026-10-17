# DEEP-GUARD Final Demonstration Guide

This guide covers the currently working local image-analysis demonstration. Keep the current checkpoint and published benchmark unchanged for this demo; planned model improvements are a separate experiment.

## Implemented status at demo time

- **Image:** trained EfficientNet-B0 baseline, FastAPI inference endpoint, and React dashboard are connected.
- **Video:** experimental/untrained. The current dashboard upload accepts images only; do not present video detection as operational.
- **Audio:** experimental/untrained. Do not present audio detection as operational.
- **Multimodal fusion:** architecture/API contract only; end-to-end evaluation is pending.
- **Pixel-level Grad-CAM:** not included in the current inference response.

## Start locally on Windows

Prerequisites: Python virtual environment at `.venv`, Node.js/npm, and a downloaded best checkpoint at `artifacts/image_training_best.pt`. The checkpoint is available in the public [training-checkpoints GitHub Release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints).

### 1. Start the backend

From the repository root in PowerShell:

```powershell
$env:DEEP_GUARD_IMAGE_CHECKPOINT = (Resolve-Path ".\\artifacts\\image_training_best.pt").Path
.\\.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal running. Confirm the service is healthy at <http://127.0.0.1:8000/api/v1/health>. The response should include `"status":"ok"`.

### 2. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open the local Vite URL printed in the terminal, usually <http://localhost:5173>.

### 3. Run automated checks

From the repository root:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
cd frontend
npm ci
npm run build
```

The repository's GitHub Actions workflow also runs Python checks and builds the frontend. Check the current status at <https://github.com/ABHILATTHE6/DEEP-GUARD/actions>.

## Examiner demo sequence

1. **Introduce the system.** DEEP-GUARD is a media-authenticity assessment research prototype. It returns model scores and a tentative verdict; it does not prove authorship or provenance.
2. **Show the dashboard and API connection.** Point out the upload panel, result/evidence panel, model registry, and recent local analysis history.
3. **Upload a sample image.** Click *Run detection*. Explain the returned verdict, AI/real class scores, selected-class score, model name, and evidence text. Do not claim the score is a calibrated probability of real-world authenticity.
4. **Demonstrate uncertainty and limitations.** Mention that individual AI-generated images can be classified as real. The current model should be treated as a baseline, not a definitive detector.
5. **Show model availability.** The image detector is trained; video/audio remain experimental/untrained; multimodal fusion is planned/not yet validated.
6. **Show evaluation evidence.** Open the [human-readable baseline report](../reports/image_evaluation_tiny_genimage.md) and [dataset manifest](../reports/dataset_manifest.json). Explain that the metrics belong to the 3,000-per-class Tiny-GenImage baseline, not the earlier mixed V3 experiment.
7. **Show reproducibility.** Point to the training notebook, scripts, report JSON, manifest, and checkpoint release in GitHub.

## Current measured image baseline

Held-out Tiny-GenImage test results:

| Metric | Result |
|---|---:|
| Test samples | 800 |
| Argmax accuracy | 85.50% |
| Accuracy with calibrated threshold | 80.88% |
| AI-generated precision | 94.27% |
| AI-generated recall | 65.75% |
| AI-generated F1 | 77.47% |
| Real-image recall | 96.00% |
| Real-image false-positive rate | 4.00% |
| ROC-AUC (AI-generated positive class) | 93.11% |

The real-probability threshold (0.0911) was calibrated from validation data to target a real-image false-positive rate of 3%. The held-out test false-positive rate was 4%, so the target is not guaranteed on new samples. At this threshold, 137 of the 400 AI-generated test images were classified as real.

These results describe this specific dataset and split. They do not establish performance on unseen generators, real-world uploads, image transformations, or different domains. The baseline has not yet been independently reproduced from a fresh runtime.

## Before submission

- [ ] Confirm both local services start from a clean terminal.
- [ ] Run the Python tests and frontend production build.
- [ ] Verify the GitHub Actions run is green.
- [ ] Capture dashboard screenshots with a representative image and an API result.
- [ ] Include the evaluation report and dataset manifest in the project report.
- [ ] Clearly identify video/audio and multimodal fusion as experimental/planned.
- [ ] Keep the current checkpoint and reported metrics separate from any future retrained model.
