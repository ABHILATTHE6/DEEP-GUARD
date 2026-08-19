# DEEP-Guard

## Multimodal AI-Generated Media & Deepfake Detection

DEEP-Guard is a research and engineering platform for assessing the authenticity of digital media across **images, videos, and audio** using trained deep-learning models.

> **Important:** DEEP-Guard provides an authenticity assessment, not an absolute guarantee. Detectors can fail on unseen generators, transformations, compression, adversarial manipulation, and distribution shifts.

### Vision

- Detect likely AI-generated images.
- Detect manipulated and deepfake videos.
- Detect synthetic or cloned speech/audio.
- Produce calibrated confidence with an `UNCERTAIN` outcome when evidence is insufficient.
- Track model and dataset provenance for predictions.
- Expose inference through a REST API and web application.

### Planned architecture

```text
Media Upload -> Modality Detection -> Image / Video / Audio Models
                                      |
                                      v
                              Multimodal Fusion
                                      |
                                      v
                       REAL / AI_GENERATED / UNCERTAIN
                                      |
                           Confidence + Explanation
```

### Current status

**M0 — Repository Foundation**

| Phase | Objective | Status |
|---|---|---|
| M0 | Repository foundation | In progress |
| M1 | Dataset infrastructure | Planned |
| M2 | Image detection baseline | Planned |
| M3 | Video deepfake detection | Planned |
| M4 | Audio synthetic-media detection | Planned |
| M5 | Multimodal fusion | Planned |
| M6 | REST API | Planned |
| M7 | Web dashboard | Planned |
| M8 | Explainability and calibration | Planned |
| M9 | Testing and CI/CD | Planned |
| M10 | Deployment and release | Planned |

### Technology direction

**ML:** Python, PyTorch, torchvision, Hugging Face Transformers, scikit-learn, OpenCV, librosa

**Backend:** FastAPI, Pydantic, Uvicorn

**Frontend:** React, TypeScript, Vite

**Engineering:** Docker, GitHub Actions, pytest, Ruff, mypy

### Repository structure

```text
DEEP-GUARD/
├── backend/       # FastAPI service
├── frontend/      # React/TypeScript application
├── ml/            # Training, evaluation and inference
├── datasets/      # Dataset registry and manifests; no raw datasets
├── models/        # Model metadata and release artifacts
├── notebooks/     # Research and exploration
├── scripts/       # Reproducible utilities
├── tests/         # Unit and integration tests
├── docs/          # Architecture and methodology
└── artifacts/     # Generated local artifacts
```

### Dataset policy

Large datasets will not be committed to Git. The repository will contain dataset documentation, provenance, manifests, preprocessing instructions, and references to official sources while respecting licenses and access requirements.

### Evaluation

Models will be evaluated with task-appropriate metrics including accuracy, precision, recall, F1, ROC-AUC, PR-AUC where appropriate, confusion matrices, calibration/error analysis, and unseen-generator or cross-dataset testing where feasible.

### Responsible AI

DEEP-Guard is an analysis and research tool. A model prediction should not be treated as definitive proof of authenticity or fraud. Human review and source/context verification remain important for high-impact decisions.

### Roadmap

1. Establish reproducible project infrastructure.
2. Build dataset registry and validation tooling.
3. Train and evaluate an image detection baseline.
4. Extend to video frame and temporal analysis.
5. Add synthetic-audio detection.
6. Develop multimodal score fusion and calibration.
7. Expose inference through FastAPI.
8. Build the DEEP-Guard analysis dashboard.
9. Add explainability, monitoring, testing, and CI/CD.
10. Package a reproducible release.

### Author

**Abhishek Latthe** — [@ABHILATTHE6](https://github.com/ABHILATTHE6)
