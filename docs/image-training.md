# M2.1 — Image training and evaluation

## Objective
Train the DEEP-Guard image baseline on an approved real/synthetic dataset and report reproducible metrics without overstating performance.

## Local workflow

1. Prepare data using the layout in `ml/image/data/README.md`.
2. Confirm the dataset is registered and its license permits the intended research use.
3. Train:

```bash
python -m ml.image.training --data path/to/data --epochs 5 --seed 42
```

4. Evaluate:

```bash
python -m ml.image.evaluate --data path/to/data/test --checkpoint artifacts/image-baseline.pt
```

## Metrics

The evaluator records accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and a per-class classification report.

## What we will not claim yet

No accuracy or benchmark number is considered a DEEP-Guard result until training has been run on a documented dataset and the resulting artifacts have been reviewed. The model must also be evaluated on data that is not used for training, and preferably on an unseen generator or dataset to assess generalization.
