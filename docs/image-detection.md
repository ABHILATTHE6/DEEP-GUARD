# Image Detection Baseline

M2 introduces the first DEEP-Guard image detector as a transfer-learning baseline.

## Model

The baseline uses **EfficientNet-B0** with ImageNet initialization and a two-class classification head:

- `real`
- `ai_generated`

The pretrained backbone is a starting point, not a final benchmark. Future work will compare stronger vision transformers and cross-dataset generalization.

## Dataset layout

Training expects a standard `ImageFolder` structure:

```text
train/
├── real/
└── ai_generated/

validation/
├── real/
└── ai_generated/
```

Raw datasets are intentionally excluded from Git. Use the dataset registry and manifests to document provenance and approved local paths.

## Training

```bash
python scripts/train_image.py \
  --train data/image/train \
  --val data/image/validation \
  --epochs 5 \
  --batch-size 32 \
  --output artifacts/image_baseline.pt
```

## Inference

```python
from ml.image.inference import predict

result = predict("sample.jpg", "artifacts/image_baseline.pt")
print(result)
```

## Evaluation requirements

A reported model result should include the dataset split, class balance, preprocessing, seed, model version, accuracy, precision, recall, F1, ROC-AUC where applicable, confusion matrix, and limitations. Cross-dataset or unseen-generator evaluation should be added before making strong claims about generalization.
