# Image Training and Checkpoint Recovery

This guide describes the repository's resumable image training script. The dataset preparation pipeline is still a separate step: before training, supply the expected folder structure shown below.

## Fast dataset recovery (public baseline)

If the previous Colab temporary files disappeared, this repository now includes a streaming preparation script using the public [TheKernel01/Tiny-GenImage dataset on Hugging Face](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage). Its dataset card lists the license as CC-BY-NC-SA 4.0; review the current source terms and cite the original dataset authors in the project report.

Run this in a fresh Colab runtime from the repository root:

```bash
python -m pip install "datasets>=3,<5" "pillow>=10,<12" "scikit-learn>=1.5,<2"
python scripts/prepare_tiny_genimage_hf.py \
  --output /content/deepguard-tiny \
  --train-per-class 3000 \
  --eval-per-class 400 \
  --buffer-size 4000 \
  --seed 42
```

The deadline-focused preparation preset streams 3,000 images per class to train and 400 per class each to validation and test. The source training split supplies train; the source validation split is divided into separate validation and test samples. The public source is listed as 8.36 GB overall, so initial preparation can still require significant network transfer. To increase sample counts, adjust these values after the first end-to-end run succeeds. The script writes `dataset_manifest.json` and can replay a seeded stream to continue a partial preparation. It may still need to download several gigabytes; do not disconnect it before it completes.

**This is a fast, reproducible Tiny-GenImage baseline, not the earlier mixed Tiny-GenImage + COCO + Nature V3 dataset.** If this fallback is used for the final demo, label its model and metrics as the Tiny-GenImage baseline. Do not compare its result to previous mixed-dataset metrics as though they came from the same test set.

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

The fastest supported route is the [public Colab notebook](https://colab.research.google.com/github/ABHILATTHE6/DEEP-GUARD/blob/main/notebooks/DEEP_GUARD_Colab_Training.ipynb), which runs the preparation, checkpoint recovery, training, evaluation, and report publication steps in order. Its deadline-first preset trains with 3,000 images per class and 5 epochs; increase to 10 epochs or a larger sample count only if time permits.

For a manually prepared dataset following the layout above, from the repository root run:

```bash
python -m pip install -e ".[ml,dev]"
python scripts/train_image.py \
  --train /content/deepguard-tiny/train \
  --val /content/deepguard-tiny/val \
  --epochs 5 \
  --batch-size 64 \
  --target-real-fpr 0.03 \
  --output artifacts/image_training_last.pt \
  --best-output artifacts/image_training_best.pt
```

The epoch count is the **target total epoch**. For example, resuming at epoch 4 with `--epochs 10` continues from epoch 5 through epoch 10.

The script saves two files after epochs complete: the most recent resumable checkpoint and the best checkpoint by AI-positive validation ROC-AUC. Each local checkpoint write is atomic. Local files in Colab's `/content` directory are still temporary until uploaded elsewhere.

## 3. Make checkpoints persistent on GitHub

The repository includes `scripts/github_release_checkpoint.py`. It creates/uses a public prerelease named by the tag `training-checkpoints` and stores checkpoint files as Release assets rather than ordinary Git commits. On a public repository, those assets are public.

For the account that owns the repository:

1. Create a fine-grained personal access token limited to this repository, granting **Contents: read and write**. Do not paste the token into a notebook cell or commit it.
2. Provide it to the runtime as `GH_TOKEN`. You may store it in Colab Secrets or use the public notebook's hidden input prompt.
3. If using Colab Secrets, read the secret into the process environment before running training:

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

GitHub Release sync preserves the training checkpoint, including model weights, optimizer state, scaler state, completed epoch, best validation AUC, calibrated decision threshold, class mapping, and random-number generator state. It does **not** preserve the dataset directory from Colab. The repository now has a scripted download/preparation path for the Tiny-GenImage fallback, but it still needs to be executed successfully from a clean runtime before the full workflow can be considered verified.

A checkpoint also does not guarantee bit-for-bit identical training after a restart, since machine/software versions and data-loader worker behavior can vary. Record the dataset version, split manifest, environment, and seed with the experiment.

## Phase 2: diverse-generator experiment

For a separate experiment aimed at generalization to more generators, use the [OpenFake experiment notebook](../notebooks/DEEP_GUARD_OpenFake_Experiment.ipynb) and [experiment guide](OPENFAKE_EXPERIMENT.md). It keeps the current Tiny-GenImage checkpoint/report unchanged, evaluates both checkpoints on the same held-out OpenFake OOD and in-the-wild test subsets, and does not upload the candidate checkpoint to the public release. Review dataset and generator-specific terms first; the candidate weights should not be redistributed until licensing is cleared.

## Current status

The resume and release-sync code is committed, but the first end-to-end publish/download/resume cycle and the clean-runtime dataset reconstruction have not yet been verified. Do not treat the project as fully reset-proof until those checks pass and a real evaluation report has been recorded.
