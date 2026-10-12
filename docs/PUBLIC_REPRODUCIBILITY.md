# Public Reproducibility and GitHub-First Workflow

This document defines how DEEP-GUARD should be organized so that a new contributor can discover the project, reproduce experiments, and run the application from the public GitHub repository without depending on one person's temporary Colab filesystem.

## Goals

- Keep source code, setup instructions, training/evaluation scripts, configuration, tests, experiment metadata, and reports in this public repository.
- Make dataset preparation deterministic and documented, with source URLs, licenses, expected directory structure, and integrity checks.
- Make GPU training resumable from saved checkpoints rather than relying on ephemeral notebook storage.
- Publish model weights only when the license and size/storage method permit it, together with their version, SHA-256 checksum, training configuration, and evaluation report.
- Keep the public app and API configuration safe: never commit tokens, credentials, private user media, or secret environment files.
- Publish measured results only; do not describe unrun experiments as completed.

## What belongs where

| Artifact | Canonical home | Important notes |
|---|---|---|
| Application source, frontend, backend, tests | This Git repository | Versioned in Git |
| Training, evaluation, dataset-preparation scripts | This Git repository | Must be runnable from a clean environment |
| Notebook entry points | This Git repository | Notebooks should invoke repository scripts rather than contain irreplaceable logic |
| Dataset metadata, attribution, split manifests | This Git repository | Include the license/source and the exact preparation procedure |
| Raw images / videos / audio | Dataset's approved source or an appropriate dataset host | Do not re-upload data unless license/terms explicitly permit redistribution |
| Model checkpoints | Release asset or an approved model/artifact host | Do not place large binary files in ordinary Git history |
| Metrics, plots, model card, checksums | This Git repository | Only include results produced by a recorded run |
| Secrets / API keys | GitHub Actions secrets or runtime secrets | Never commit these |
| CI tests and frontend build | GitHub Actions | Suitable for normal CPU-based validation |
| GPU training | GPU-enabled runner or external notebook/compute runtime | The code and workflow can be public and reproducible even if the actual GPU is outside standard GitHub runners |

## Public reproduction path

1. Clone this repository.
2. Read the dataset license and source instructions before downloading anything.
3. Run the dataset preparation script. It must fail clearly if a source file is missing, verify the expected class directories and sample counts, and print the resulting split summary.
4. Run a short CPU smoke test to verify imports, model construction, and a tiny batch.
5. Start a full training run on an available GPU.
6. Save a resume checkpoint periodically. A resume checkpoint should contain model state, optimizer state, completed epoch, best validation metric, random-number-generator state where practical, class mapping, config, and dataset version/hash.
7. Calibrate thresholds with validation data only. Do not tune thresholds on the test split.
8. Evaluate the selected checkpoint on the held-out test split and save metrics, confusion matrix, ROC/PR data or plots where appropriate, and a run metadata file.
9. Publish the checkpoint as a versioned public artifact only if redistribution is permitted. Add its SHA-256 hash and the matching evaluation report to the repository.
10. Run the application using an explicit checkpoint path. If no checkpoint is installed, the API must report that the model is unavailable rather than implying a real assessment was performed.

## Persistence rules

A Colab runtime's local `/content` directory is temporary. Never treat it as the sole copy of a dataset, checkpoint, or important log. A notebook should be able to recreate temporary files from documented sources and restore a resumable checkpoint from persistent storage.

Do not push a dataset or checkpoint to ordinary Git just to make it persistent. GitHub blocks regular Git files above 100 MiB and recommends keeping repositories reasonably small. Git LFS, GitHub Release assets, and external artifact hosting each have separate limits, costs, and usage terms. Pick the storage method after checking the size, public-download requirements, and dataset/model licenses.

## Hosting boundaries

- GitHub is the canonical public source repository and can run CI workflows.
- GitHub Pages can host a static frontend; it cannot run the FastAPI Python backend.
- The FastAPI service must run on a separate supported host or on infrastructure you manage.
- Standard GitHub Actions runners are useful for tests, linting, builds, and small CPU smoke tests. GPU training requires an available GPU runner or an external GPU runtime; do not assume a standard runner includes CUDA.
- A public Colab notebook can be launched from the repository, but its runtime and local files are still temporary. The notebook must restore data/checkpoints at the beginning and save checkpoints during training.

## Current status

This document is the target workflow, not evidence that every step has already been completed. Before calling the project fully reproducible, the repository still needs a verified dataset download/preparation path, a checked-in GPU notebook or step-by-step runner, tested resume support in the training script, and a published evaluation report for the final checkpoint. Previous training metrics must not be presented as final if their probability/label orientation or data split has not been verified.

## Acceptance checklist

- [ ] Fresh clone can install dependencies and pass tests.
- [ ] Public instructions state the actual supported environment.
- [ ] Dataset sources, licenses, split definitions, and preparation are recorded.
- [ ] Dataset preparation validates paths, classes, counts, and split integrity.
- [ ] Training is deterministic as far as practical and records config, seed, and dataset metadata.
- [ ] Training can resume from a persistent checkpoint after a runtime restart.
- [ ] Validation is used for checkpoint selection and threshold calibration; test set is held out.
- [ ] Evaluation script produces metrics with verified class/probability orientation.
- [ ] Published weights have version, checksum, matching config, and evaluation report.
- [ ] GitHub Actions runs tests and frontend build on pull requests/pushes.
- [ ] Public demo has a deployment URL and documents API hosting.
- [ ] README distinguishes implemented, experimentally validated, and planned features.
