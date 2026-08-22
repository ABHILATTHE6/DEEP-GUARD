# DEEP-Guard Image Datasets

This directory contains dataset metadata and preparation documentation only. Raw datasets are intentionally excluded from Git.

## M2.2 strategy

Separate **training data** from **generalization data**. Do not mix the same source/generator across train and test splits.

### Primary candidate

**CIFAKE** is a practical image-level baseline for real-vs-AI-generated image classification. Before training, record its exact source/version, license or terms, class balance, preprocessing, and split.

### Generalization

For a stronger experiment, add an independent test set containing images from generators not represented in training. Report this separately from the in-distribution test score.

### Future video data

FaceForensics++ is reserved primarily for the later video/deepfake phase. Its official project describes 1,000 original video sequences manipulated using four methods and provides access under its terms.

## Local layout

```text
datasets/local/image/
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

## Reproducibility requirements

Before training, record dataset name/version, source/access date, license/terms, image counts per class, formats/dimensions, split strategy and seed, preprocessing/augmentation configuration, and checksums/manifests where practical.

Never commit downloaded dataset files or credentials.