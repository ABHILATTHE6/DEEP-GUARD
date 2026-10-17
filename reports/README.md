# Evaluation Reports

This directory contains versioned experiment reports and dataset metadata. Raw media datasets and large model checkpoints are stored separately; they do not belong in ordinary Git history.

## Published evaluation

- [Tiny-GenImage image baseline — human-readable report](image_evaluation_tiny_genimage.md)
- [Tiny-GenImage image baseline — machine-readable JSON](image_evaluation.json)

The current report describes the deadline-focused Tiny-GenImage baseline trained for five epochs using 3,000 images per class. It is not the previous mixed COCO + Nature + Tiny-GenImage V3 experiment.

## Next reproducibility step

Run the public [Colab training notebook](https://colab.research.google.com/github/ABHILATTHE6/DEEP-GUARD/blob/main/notebooks/DEEP_GUARD_Colab_Training.ipynb) through its final publication cell. That cell updates the JSON report with checkpoint and dataset-manifest checksums and commits the actual dataset manifest. Do not publish placeholder performance metrics.
