# DEEP-Guard

## Multimodal AI-Generated Media & Deepfake Detection

DEEP-Guard is a research and engineering platform for assessing the authenticity of digital media across **images, videos, and audio** using task-specific deep-learning pipelines.

> **Important:** DEEP-Guard provides an authenticity assessment, not an absolute guarantee. Detectors can fail on unseen generators, transformations, compression, adversarial manipulation, and distribution shifts.

## Current implementation status

| Capability | Status |
|---|---|
| Repository + project structure | ✅ Implemented |
| Image EfficientNet-B0 training/evaluation pipeline | ✅ Trained baseline and held-out evaluation published; clean-runtime reproduction pending |
| Image inference API | ✅ Implemented; requires a compatible trained checkpoint |
| React examiner dashboard | ✅ Implemented |
| Video screening | ✅ Dashboard/API accepts video; scores up to 8 sampled frames with the image checkpoint; experimental, no temporal model or video benchmark |
| Audio screening | ✅ Dashboard/API supports an external pretrained Wav2Vec2 speech detector; first use downloads ~1.26 GB; experimental, no DEEP-GUARD audio benchmark |
| Multimodal evidence fusion | ⏳ Architecture planned; end-to-end evaluation pending |
| Grad-CAM utility | ⏳ Utility exists; not enabled in current inference response |
| CI workflow | ✅ Python tests, frontend build, container startup, and image-inference smoke tests pass |
| Trained video checkpoint | ⏳ Requires dataset + training run |
| Trained audio checkpoint | ⏳ Requires dataset + training run |
| Cross-generator benchmark | ⏳ Requires evaluation data |
| Robustness benchmark | ⏳ Requires evaluation data |

The current operational image path uses the trained EfficientNet-B0 baseline. Video requests run an **experimental frame-aggregation screen** using up to eight sampled frames; this is not a temporal video model and has not been benchmarked on a dedicated video dataset. Audio requests use an external pretrained Wav2Vec2 speech detector (English-focused; about 1.26 GB of weights downloaded on first use); this external model has not been independently benchmarked by DEEP-GUARD. These two modalities are functional prototypes, not DEEP-GUARD-trained/validated detectors. The best image checkpoint is published in the `training-checkpoints` GitHub Release, and measured Tiny-GenImage baseline results and a dataset manifest are published under `reports/`. The repository does not fabricate benchmark accuracy.

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
        EfficientNet      sampled frames   Wav2Vec2 speech
        + Grad-CAM        + image model    classifier

             |                |                |
             +----------------+----------------+
                              |
                       Evidence Fusion
                              |
                +-------------+-------------+
                |             |             |
             REAL       AI-GENERATED     UNCERTAIN
```

## Remaining experimental work

### 1. Prepare licensed datasets

Keep raw datasets outside Git. Register each dataset with its source, license, class definitions, generator/source information, and version. Create independent train/validation/test splits and prevent near-duplicate leakage.

### 2. Improve the image baseline (Phase 2)

The current five-epoch Tiny-GenImage baseline, best checkpoint, dataset manifest, and held-out evaluation are already published. Keep this baseline intact. The new [OpenFake generalization experiment](notebooks/DEEP_GUARD_OpenFake_Experiment.ipynb) trains a separate candidate with a broader generator distribution and compares both models on the same held-out OOD and in-the-wild test samples. It does not overwrite the existing checkpoint or publish new weights automatically. Review the dataset's non-commercial and generator-specific license terms before use or redistribution; see [Phase 2 experiment instructions](docs/OPENFAKE_EXPERIMENT.md).

### 3. Video and audio are experimental

The API currently reports video and audio as uncertain/experimental rather than presenting an unverified detection result. Frame analysis and audio feature extraction can be developed from the existing prototypes, but modality-specific training data, model checkpoints, integration, and held-out evaluation remain necessary before claiming video/audio detection is operational.

### 4. Train the audio model

Use real and synthetic/cloned speech data with compatible licenses. Train a CNN/Transformer over Mel-spectrograms, save the checkpoint, then connect it to `ml/media/audio.py`.

### 5. Validate fusion

Evaluate image-only, video-only, audio-only, and multimodal performance. Calibrate thresholds on validation data and report disagreement/uncertainty instead of forcing a binary decision.

### 6. Robustness and generalization

Test JPEG compression, resizing, blur, noise, screenshots, transcoding, and at least one generator/domain not represented in training. Report the performance drop rather than only the best-case score.

### 7. Final examiner demonstration

Use the current dashboard to upload and analyze an image, show the model scores and evidence, and open the model-availability panel. The current dashboard accepts images only. Video/audio detection remains experimental/untrained, multimodal fusion is not end-to-end validated, and Grad-CAM is not enabled in the current inference response. Use measured results only.

## Local run

For quick local startup on Windows, see [Local Demo Quick Start](docs/LOCAL_DEMO_QUICKSTART.md) and `scripts/start_demo.ps1`. For modality behavior and the external audio model, see [Multimodal Demo Notes](docs/MULTIMODAL_PHASE.md). Public hosting is optional; the current presentation path runs locally. For the full examiner script, see [Final Demonstration Guide](docs/FINAL_DEMO.md); for presentation talking points and viva preparation, see [Presentation and Viva Guide](docs/PRESENTATION_AND_VIVA.md).

Start the backend from the repository root after downloading `artifacts/image_training_best.pt` from the [training-checkpoints release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints):

```powershell
$env:DEEP_GUARD_IMAGE_CHECKPOINT = (Resolve-Path ".\artifacts\image_training_best.pt").Path
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm ci
npm run dev
```

Verify the backend at <http://127.0.0.1:8000/api/v1/health>. The React dashboard is served at the Vite URL printed by `npm run dev`, usually <http://localhost:5173>.

For the local Windows demo, use the [Local Demo Quick Start](docs/LOCAL_DEMO_QUICKSTART.md), which launches the backend and dashboard without cloud setup. Public hosting is optional; if needed later, see [Cloud Run Deployment](docs/CLOUD_RUN_DEPLOYMENT.md), which requires Google Cloud billing setup.

## Evaluation

Models should be evaluated with task-appropriate metrics including accuracy, precision, recall, F1, ROC-AUC, PR-AUC where appropriate, confusion matrices, calibration/error analysis, and unseen-generator or cross-dataset testing where feasible.

## Why DEEP-Guard is different

DEEP-Guard is not a general-purpose chatbot. It is a specialized authenticity-assessment system that combines modality-specific deep-learning models, explainability, uncertainty-aware evidence fusion, robustness testing, and media provenance signals. The project is designed to expose evidence and limitations rather than present a single unexplained binary answer.

## Responsible AI

A model prediction should not be treated as definitive proof of authenticity or fraud. Human review and source/context verification remain important for high-impact decisions.

## Author

**Abhishek Latthe** — [@ABHILATTHE6](https://github.com/ABHILATTHE6)

## Public reproducibility and persistence

The public GitHub repository is the canonical home for source code, scripts, tests, notebooks, configuration, dataset preparation instructions, and verified evaluation reports. See [Public Reproducibility and GitHub-First Workflow](docs/PUBLIC_REPRODUCIBILITY.md) for the intended end-to-end workflow and current completion checklist.

**Important:** Colab's local `/content` storage is temporary. Dataset files and training checkpoints must be restored from a documented source and saved to persistent storage during a run. Large datasets and model checkpoints should not be committed as ordinary Git files; publish only data that may legally be redistributed, and use an appropriate release/artifact location for large binaries.

GitHub Actions currently provides the repository's CI checks. The standard CI workflow validates the project and builds the frontend; it is not the GPU training environment. GPU training must run on a GPU-enabled runner or external GPU runtime until the project configures suitable GPU infrastructure. The training code and experiment instructions should remain versioned and publicly accessible here.

For image training commands and checkpoint recovery from GitHub Releases, see [Image Training and Checkpoint Recovery](docs/TRAINING.md). End-to-end release upload/download/resume and a clean-runtime dataset rebuild still need independent verification.

## Fast final-phase workflow

For the deadline-focused reproducible image baseline, open the [DEEP-GUARD Colab Training notebook](https://colab.research.google.com/github/ABHILATTHE6/DEEP-GUARD/blob/main/notebooks/DEEP_GUARD_Colab_Training.ipynb). It prepares a public Tiny-GenImage baseline, can restore/publish checkpoints through GitHub Releases, evaluates a held-out test split, and publishes the small report/manifest to `reports/`. Use a GPU runtime and a fine-grained GitHub token with repository Contents read/write permission if checkpoint/report publishing is required.

The fallback dataset protocol is described in [docs/TRAINING.md](docs/TRAINING.md). It is **not** the previous mixed COCO + Nature + Tiny-GenImage V3 experiment; label its metrics accordingly. The current baseline's measured results, manifest, and checksums are already published; a full clean-runtime reproduction remains pending.

See [reports/README.md](reports/README.md) for the state of final evaluation artifacts.

## Latest measured image baseline

The held-out Tiny-GenImage baseline evaluation is published: [human-readable report](reports/image_evaluation_tiny_genimage.md), [machine-readable JSON](reports/image_evaluation.json), and [dataset manifest](reports/dataset_manifest.json). The published JSON records SHA-256 checksums for the dataset manifest and best checkpoint. These results are for the 3,000-per-class quick baseline only, not the earlier mixed V3 data protocol. A full clean-runtime reproduction still needs verification.
