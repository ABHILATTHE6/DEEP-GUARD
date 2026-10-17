# DEEP-GUARD Image Baseline — Evaluation Report

**Evaluation date:** 2026-10-17  
**Model:** `deepguard-image-efficientnet-b0`  
**Checkpoint:** GitHub Release `training-checkpoints` → `image-training-best.pt`  
**Training epoch:** 5  
**Dataset:** [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage)

> This is the deadline-focused Tiny-GenImage baseline, not the earlier mixed COCO + Nature + Tiny-GenImage V3 experiment. Its metrics should not be presented as results from that different dataset protocol.

## Dataset splits

| Split | AI-generated | Real | Total |
|---|---:|---:|---:|
| Train | 3,000 | 3,000 | 6,000 |
| Validation | 400 | 400 | 800 |
| Test | 400 | 400 | 800 |
| **Total** | **3,800** | **3,800** | **7,600** |

The source training split supplies training images; the source validation split is divided into disjoint validation and test subsets. Validation is used for decision-threshold calibration; the test split is used for the final metrics below.

## Held-out test results

| Metric | Result |
|---|---:|
| Argmax accuracy | **85.50%** |
| Accuracy using calibrated decision threshold | **80.88%** |
| AI-generated precision | **94.27%** |
| AI-generated recall | **65.75%** |
| AI-generated F1 | **77.47%** |
| Real-image recall | **96.00%** |
| Real-image false-positive rate | **4.00%** |
| ROC-AUC (AI-generated is positive class) | **93.11%** |

### Confusion matrix

Rows are actual classes and columns are predicted classes, in the order **real, AI-generated**:

| Actual  Predicted | Real | AI-generated |
|---|---:|---:|
| Real | 384 | 16 |
| AI-generated | 137 | 263 |

Of 400 AI-generated test images, 263 were detected by the calibrated threshold and 137 were classified as real. Of 400 real images, 16 were classified as AI-generated.

## Decision threshold

The evaluation script calibrates a threshold on the validation split only, targeting a real-image false-positive rate no greater than 3%. The selected threshold was **P(real) = 0.0911**. On validation, AI recall was 64.75% and real-image FPR was 2.75%. On the held-out test split, real-image FPR was 4.00%, which is higher than the validation target; this illustrates that calibration does not guarantee identical behavior on new data.

Argmax accuracy and calibrated-threshold accuracy differ because they use different decision rules. They should be reported separately, as above.

## Interpretation and limitations

The model shows useful separation on this particular held-out Tiny-GenImage split (ROC-AUC 0.9311), with high AI precision (94.27%) and real-image recall (96.00%) at the calibrated operating point. However, AI recall is 65.75%, so about one third of AI-generated samples in this test split were missed at that threshold. The argmax operating point provides 85.50% accuracy but has its own precision/recall trade-offs, which should be examined before selecting an application operating point.

These scores describe only this dataset protocol and split. They are not proof of media provenance and do not establish performance on unseen generators, real-world photos, post-processing, compression, or cross-dataset samples. The dataset license and attribution terms must be respected; raw dataset images are not included in this repository.

The machine-readable metrics are in [`image_evaluation.json`](image_evaluation.json). The dataset manifest will be published after the notebook's final report-publication cell completes.
