# Image training data layout

DEEP-Guard does not store raw datasets in Git. Download data according to the dataset registry and place it locally using this layout:

```text
<data-root>/
├── train/
│   ├── ai_generated/
│   └── real/
├── val/
│   ├── ai_generated/
│   └── real/
└── test/
    ├── ai_generated/
    └── real/
```

The training and evaluation scripts use `torchvision.datasets.ImageFolder` and require the same class names in every split.

## Reproducibility requirements

Record the dataset identifier, source, license/usage terms, preprocessing, split strategy, random seed, model version, and environment in the experiment notes. Never commit private, restricted, or large raw media to the repository.
