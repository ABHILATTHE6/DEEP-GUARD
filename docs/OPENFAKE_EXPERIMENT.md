# Phase 2 — Diverse Image Model Experiment

## Goal

Improve generalization of the existing EfficientNet-B0 image detector without overwriting the published Tiny-GenImage baseline. This experiment uses a different dataset with source-defined train, validation, out-of-distribution (OOD) test, and in-the-wild test sets.

## Dataset choice

Use [OpenFake on Hugging Face](https://huggingface.co/datasets/ComplexDataLab/OpenFake). Its dataset card describes a diverse set of generators and separates `core/test` from training generators, with different real-image sources, and provides a separate `reddit/test` in-the-wild evaluation. See the [dataset card and licensing notes](https://huggingface.co/datasets/ComplexDataLab/OpenFake).

**Licensing caution:** the dataset page's top-level metadata reports CC-BY-NC-4.0, and its card explains that some generator subsets have additional restrictions. Treat this as an academic/non-commercial experiment, inspect the current license/card and per-generator terms, do not redistribute raw images, and do not publish a new checkpoint until its redistribution terms are clear. The existing checkpoint and Tiny-GenImage report remain unchanged.

## Sampling preset

The preparation script samples the following balanced per-class counts without using the OOD/wild test sets for training:

| Output | Source split | Per class | Purpose |
|---|---|---:|---|
| `train` | `core/train` | 4,000 | Training (8,000 images total) |
| `val` | `core/validation` | 500 | Validation and threshold calibration |
| `test_ood` | `core/test` | 800 | Main held-out OOD test (1,600 images total) |
| `test_wild` | `reddit/test` | 300 | Separate in-the-wild test (600 images total) |

Total local sample target: 11,200 images. The script records the source model names encountered and writes `dataset_manifest.json`. It downloads only the streamed shards needed to find the requested samples, but transfer time depends on dataset shard layout and Colab networking.

## Run in Google Colab

Open a new notebook from the repo file `notebooks/DEEP_GUARD_OpenFake_Experiment.ipynb`: [launch the experiment notebook](https://colab.research.google.com/github/ABHILATTHE6/DEEP-GUARD/blob/main/notebooks/DEEP_GUARD_OpenFake_Experiment.ipynb). Select a GPU runtime before running the cells.

The notebook:

1. Clones/pulls the current public repository and installs dependencies.
2. Checks that a GPU is available.
3. Downloads the existing baseline checkpoint to a separate path.
4. Samples the OpenFake data into `/content/deepguard-openfake`.
5. Evaluates the **existing baseline** on the same OpenFake OOD and wild test sets.
6. Trains a candidate checkpoint to `/content/deepguard-openfake-artifacts/openfake-best.pt` and the matching latest resume checkpoint—never to the baseline filename.
7. Evaluates the candidate on the exact same test samples for apples-to-apples comparison. Both checkpoints get threshold calibration from the same `core/validation` subset only; neither test split affects threshold selection.
8. Packages the candidate checkpoint, JSON metrics, and manifest into a ZIP for downloading to your computer. It does not publish the new model weights to GitHub.

## Promotion criteria

Do not replace the old checkpoint just because training loss falls. Compare baseline and candidate on the **same** `test_ood` and `test_wild` samples, paying special attention to AI recall, ROC-AUC, real-image false-positive rate, and confusion matrices. If OOD or wild-test performance improves while unacceptable false positives are not introduced, save the candidate as a separate version and test it locally. If it fails to improve, retain the baseline.

Keep test sets fixed. Do not use their scores to adjust hyperparameters or thresholds. Use validation data for those choices; if you make more choices after seeing test results, reserve a fresh final test set.

## What is intentionally not done

- This does not retrain video/audio models.
- This does not overwrite `image-training-best.pt` or the published Tiny-GenImage report.
- This does not upload the OpenFake checkpoint to the public GitHub Release, due to dataset/license uncertainty.
- It does not guarantee correct classifications on every image; generalization must be measured.

## Baseline artifacts

- [Current report](../reports/image_evaluation_tiny_genimage.md)
- [Current manifest](../reports/dataset_manifest.json)
- [Current checkpoint release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints)