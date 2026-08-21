# Dataset Infrastructure

DEEP-Guard keeps dataset metadata in Git while keeping raw datasets outside the repository.

## Registry

`datasets/registry.yaml` is the source of truth for dataset identity, modality, task, provenance, access requirements, labels, and licensing notes.

## Manifests

Dataset manifests describe reproducible splits and preprocessing. They must include a schema version, source/provenance information, labels, split references, and reproducibility settings.

## Rules

- Do not commit raw datasets or restricted media.
- Do not commit credentials or private download links.
- Review official dataset terms before use.
- Keep train/validation/test splits deterministic.
- Prevent source-level leakage when source grouping information is available.
- Record preprocessing versions so experiments can be reproduced.

## Initial dataset direction

The project currently tracks video datasets such as FaceForensics++ and DF40 as planned sources. Image and audio datasets will be registered only after their provenance and access terms are reviewed.

## Validation

The Python registry validator will become part of the CI quality gate. Invalid modalities, labels, missing required metadata, and duplicate dataset IDs should fail validation before training pipelines consume the registry.
