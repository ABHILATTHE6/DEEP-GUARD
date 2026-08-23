# DEEP-Guard Image Training

This module provides the first reproducible image training/evaluation path for DEEP-Guard.

## Dataset layout

```text
data/image/
├── train/
│   ├── real/
│   └── ai_generated/
├── val/
│   ├── real/
│   └── ai_generated/
└── test/
    ├── real/
    └── ai_generated/
```

Raw datasets are intentionally not committed to Git.

## Train

```bash
python -m ml.image.training.train --data data/image --epochs 10 --batch-size 32
```

## Evaluate

```bash
python -m ml.image.evaluation.evaluate \
  --data data/image/test \
  --checkpoint artifacts/experiments/image/efficientnet-b0/best.pt
```

## Inference

```bash
python -m ml.image.inference.predict \
  path/to/image.jpg \
  --checkpoint artifacts/experiments/image/efficientnet-b0/best.pt
```

## Scientific reporting

Do not add benchmark numbers until a reproducible training run has been completed. Report the dataset/version, split policy, seed, model configuration, and test metrics together. Test performance on an independent or unseen-generator dataset before making generalization claims.
