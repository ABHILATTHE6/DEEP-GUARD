# Evaluation Reports

This directory contains versioned experiment reports and dataset metadata. Raw media datasets and large model checkpoints are stored separately; they do not belong in ordinary Git history.

## Published evaluation

- [Tiny-GenImage image baseline — human-readable report](image_evaluation_tiny_genimage.md)
- [Tiny-GenImage image baseline — machine-readable JSON](image_evaluation.json)

The current report describes the deadline-focused Tiny-GenImage baseline trained for five epochs using 3,000 images per class. It is not the previous mixed COCO + Nature + Tiny-GenImage V3 experiment.

## Dataset provenance and checksums

- [Dataset manifest](dataset_manifest.json) records the Tiny-GenImage source, split protocol, class order, and split counts.
- `image_evaluation.json` records the SHA-256 checksums for the published manifest and best checkpoint.
- The best and latest training checkpoints are stored in the [training-checkpoints GitHub Release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints).

## Remaining reproducibility step

Run the public [Colab training notebook](https://colab.research.google.com/github/ABHILATTHE6/DEEP-GUARD/blob/main/notebooks/DEEP_GUARD_Colab_Training.ipynb) from a fresh runtime to verify dataset reconstruction and checkpoint recovery end to end. Do not publish placeholder performance metrics.
