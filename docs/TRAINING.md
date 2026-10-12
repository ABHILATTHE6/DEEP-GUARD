# Image Training and Checkpoint Recovery

This guide describes the repository's resumable image training script. The dataset preparation pipeline is still a separate step: before training, supply the expected folder structure shown below.

## 1. Expected data layout

```text
DATA_ROOT/
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

Use `train` to update model weights, `val` to select checkpoints and calculate validation ROC-AUC, and keep `test` held out for a separate final evaluation. `scripts/train_image.py` trains on `train` and evaluates checkpoint selection on `val`; it does not use the test folder.

The ImageFolder class order is alphabetical: `ai_generated` is index 0 and `real` is index 1. The script explicitly validates this order and computes ROC-AUC with the AI-generated class as the positive class.

## 2. Install and run

From the repository root in Colab or another Linux environment:

```bash
python -m pip install -e ".[ml,dev]"
python scripts/train_image.py \
  --train /content/deepguard-v3/train \
  --val /content/deepguard-v3/val \
  --epochs 10 \
  --batch-size 64 \
  --output artifacts/image_training_last.pt \
  --best-output artifacts/image_training_best.pt
```

The epoch count is the **target total epoch**. For example, resuming at epoch 4 with `--epochs 10` continues from epoch 5 through epoch 10.

The script saves two files after epochs complete: the most recent resumable checkpoint and the best checkpoint by AI-positive validation ROC-AUC. Each local checkpoint write is atomic. Local files in Colab's `/content` directory are still temporary until uploaded elsewhere.

## 3. Make checkpoints persistent on GitHub

The repository includes `scripts/github_release_checkpoint.py`. It creates/uses a public prerelease named by the tag `training-checkpoints` and stores checkpoint files as Release assets rather than ordinary Git commits. On a public repository, those assets are public.

For the account that owns the repository:

1. Create a fine-grained personal access token limited to this repository, granting **Contents: read and write**. Do not paste the token into a notebook cell or commit it.
2. Store it in the notebook provider's secret manager as `GH_TOKEN`. In Google Colab, add a notebook secret named `GH_TOKEN` and grant the notebook access to it.
3. Read that secret into the process environment before running training:

```python
import os
from google.colab import userdata

token = userdata.get("GH_TOKEN")
if token:
    os.environ["GH_TOKEN"] = token
```

On the first training run, add `--publish-to-github` to the training command. After each epoch, the script uploads the latest checkpoint as `image-training-latest.pt`; whenever validation AUC improves, it uploads `image-training-best.pt`. The latest asset is replaced rather than creating a Git commit every epoch.

**Security and visibility:** everyone can download these release assets because the repository is public. The token itself must remain secret. Do not use this option for private datasets, user uploads, credentials, or artifacts that must not be public.

## 4. Restore after a runtime reset

First clone the public repository again and recreate the training environment. Restore the checkpoint files from the release:

```bash
python scripts/github_release_checkpoint.py download \
  --repo ABHILATTHE6/DEEP-GUARD \
  --tag training-checkpoints \
  --asset-name image-training-latest.pt \
  --output artifacts/image_training_last.pt

python scripts/github_release_checkpoint.py download \
  --repo ABHILATTHE6/DEEP-GUARD \
  --tag training-checkpoints \
  --asset-name image-training-best.pt \
  --output artifacts/image_training_best.pt
```

Then resume using the latest checkpoint, retaining the same dataset split:

```bash
python scripts/train_image.py \
  --train /content/deepguard-v3/train \
  --val /content/deepguard-v3/val \
  --epochs 10 \
  --batch-size 64 \
  --output artifacts/image_training_last.pt \
  --best-output artifacts/image_training_best.pt \
  --resume-from artifacts/image_training_last.pt \
  --publish-to-github
```

The download operation does not require a token when the release is public. Uploading requires the secret token described above. If the release asset does not exist yet, run a first training epoch with publishing enabled.

## 5. What this does not persist

GitHub Release sync preserves the training checkpoint, including model weights, optimizer state, scaler state, completed epoch, best validation AUC, class mapping, and random-number generator state. It does **not** preserve the dataset directory from Colab. The repository still needs a reproducible dataset download/preparation script using the chosen sources and their licenses. Until that exists and has been tested from a clean runtime, dataset reconstruction remains a manual prerequisite.

A checkpoint also does not guarantee bit-for-bit identical training after a restart, since machine/software versions and data-loader worker behavior can vary. Record the dataset version, split manifest, environment, and seed with the experiment.

## Current status

The resume and release-sync code is committed, but the first end-to-end publish/download/resume cycle and the clean-runtime dataset reconstruction have not yet been verified. Do not treat the project as fully reset-proof until those checks pass and a real evaluation report has been recorded.
