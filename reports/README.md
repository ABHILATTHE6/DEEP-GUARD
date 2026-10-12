# Evaluation Reports

This directory holds small, versioned experiment reports and dataset manifests. Raw media data and model checkpoints do not belong in ordinary Git history.

The Colab workflow can publish:

- `image_evaluation.json` — validation-calibrated operating threshold and held-out test metrics.
- `dataset_manifest.json` — source dataset, split sizes, label mapping, and preparation seed.

These files are intentionally not populated with placeholder performance metrics. They will appear after a real run completes and publishes its results. The quick recovery notebook uses TheKernel01/Tiny-GenImage and is a separate baseline protocol, not the earlier mixed V3 split.

Report key metrics with class/probability orientation, split counts, source license/attribution, threshold calibration procedure, checkpoint SHA-256, and limitations. Never describe metrics from another dataset protocol as though they came from this run.
