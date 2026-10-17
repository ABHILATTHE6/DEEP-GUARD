# DEEP-GUARD Presentation and Viva Guide

Use this guide for the college project demonstration. It describes the current project state; do not present planned or experimental features as completed.

## 1. Short project introduction

DEEP-GUARD is a media-authenticity assessment research prototype. Its current operational path is image analysis: a React dashboard sends an uploaded image to a FastAPI backend, which calls a trained EfficientNet-B0 classifier and displays the verdict, class scores, model information, and evidence text. The result is a statistical assessment, not proof of authenticity, authorship, or provenance.

## 2. Problem statement

AI-generated and manipulated media can be difficult to distinguish visually. DEEP-GUARD provides a repeatable image-classification workflow and makes its model output, benchmark, and limitations visible to the user. The application is not a guarantee that a media file is real or fake.

## 3. Objectives

- Build a usable dashboard for uploading an image and viewing its assessment.
- Provide a local inference API backed by a trained image-classification checkpoint.
- Record evaluation metrics on a held-out dataset split.
- Publish the checkpoint reference, dataset protocol, manifest, and evaluation report for transparency.
- Clearly identify functionality that is still experimental or planned.

## 4. Architecture and request flow

1. The user selects an image in the React/Vite frontend.
2. The frontend sends the image as multipart form data to `POST /api/v1/analyze`.
3. FastAPI receives the file, applies an upload-size limit, and routes it to the image analyzer.
4. The inference module loads the configured EfficientNet-B0 checkpoint, preprocesses the image, and calculates scores for `ai_generated` and `real`.
5. The API returns the verdict, top-class score, class scores, model details, and explanatory text.
6. The dashboard renders the result and stores a small recent-analysis history in the browser's local storage.

Backend health endpoint: `GET /api/v1/health`.

## 5. Model and dataset used for the published baseline

- Model: `deepguard-image-efficientnet-b0` (EfficientNet-B0).
- Training length: 5 epochs.
- Dataset: [TheKernel01/Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage).
- Training split: 3,000 AI-generated and 3,000 real images.
- Validation split: 400 AI-generated and 400 real images.
- Test split: 400 AI-generated and 400 real images.
- Total selected samples: 7,600.
- Best checkpoint: [GitHub Release asset](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints).
- Dataset counts and protocol: [dataset manifest](../reports/dataset_manifest.json).

This is the deadline-focused Tiny-GenImage baseline. It is not the earlier mixed COCO + Nature + Tiny-GenImage V3 experiment.

## 6. Published held-out test results

| Metric | Result |
|---|---:|
| Test images | 800 |
| Argmax accuracy | 85.50% |
| Accuracy at calibrated threshold | 80.88% |
| AI-generated precision | 94.27% |
| AI-generated recall | 65.75% |
| AI-generated F1 | 77.47% |
| Real-image recall | 96.00% |
| Real-image false-positive rate | 4.00% |
| ROC-AUC (AI-generated is positive class) | 93.11% |

Confusion matrix at the calibrated operating point. Rows are actual classes; columns are predicted classes in the order real, AI-generated:

| Actual class | Predicted real | Predicted AI-generated |
|---|---:|---:|
| Real | 384 | 16 |
| AI-generated | 137 | 263 |

The threshold was selected from validation data to target a real-image false-positive rate of 3%. The measured test false-positive rate was 4%, so the validation target did not carry over exactly to the test split.

Report: [human-readable evaluation](../reports/image_evaluation_tiny_genimage.md) and [machine-readable JSON](../reports/image_evaluation.json).

## 7. Important limitations to state during the demo

- The model can misclassify AI-generated images as real. In the published test split, 137 of 400 AI-generated images were classified as real at the calibrated threshold.
- A very high top-class score is the model's output score, not a guarantee of correctness or a calibrated probability of real-world authenticity.
- The benchmark is specific to this dataset and split; it does not establish performance on unseen generators, different domains, or transformed/compressed images.
- The dashboard accepts images, video, and audio. Image is the trained project baseline; video is experimental frame aggregation; audio is an external pretrained English speech classifier; multimodal fusion is not validated end to end.
- Grad-CAM heatmaps are not enabled in the current inference response.
- A fresh-runtime dataset rebuild and the full checkpoint upload/download/resume cycle still need independent verification.

## 8. Suggested live demonstration

1. Open the dashboard and point out the API-connection status.
2. Upload a sample image and run detection.
3. Explain the verdict, AI/real scores, model name, and returned evidence.
4. Open the model-availability section and show which modalities are ready versus experimental/planned.
5. Open the evaluation report and explain accuracy, precision/recall, ROC-AUC, and the confusion matrix.
6. Show the GitHub repository and checkpoint release.
7. State clearly that the system provides an assessment, not definitive proof, and that the image model remains a baseline for future improvement.

## 9. Viva questions and concise answers

**Q: What does DEEP-GUARD do?**  
A: It provides an image-authenticity assessment through a dashboard, a FastAPI service, and a trained EfficientNet-B0 image classifier. Video/audio and fusion are not yet operational as trained detectors.

**Q: Why use EfficientNet-B0?**  
A: It is a convolutional image-classification architecture that offers a practical starting point for transfer-learning experiments. The current result is a baseline, not evidence that it is the best possible architecture.

**Q: What is the role of FastAPI?**  
A: It exposes the health and media-analysis endpoints, receives the uploaded file, routes it to inference, and returns structured JSON for the frontend.

**Q: What is the difference between precision and recall?**  
A: Precision asks how many images predicted as AI-generated actually belong to that class in the test set. Recall asks how many actual AI-generated test images the detector successfully identifies. Here, AI precision is 94.27%, while AI recall is 65.75%.

**Q: Why report both argmax accuracy and calibrated-threshold accuracy?**  
A: They use different decision rules. Argmax picks the class with the larger score. The calibrated threshold was selected on validation data to target a low false-positive rate for real images. They should not be mixed as if they were the same operating point.

**Q: Does a 100% score mean an image is definitely real?**  
A: No. It is the model's top-class score for that input, not proof of authenticity or a guarantee of correctness. The model has misclassified some AI-generated images.

**Q: Why did an AI-generated example get classified as real?**  
A: The classifier can fail on individual samples and on images that differ from the patterns it learned. Improving the training data and evaluating a new checkpoint on held-out data is planned future work; changing a threshold alone cannot correct every confidently wrong prediction.

**Q: Why is the test set separate from validation?**  
A: Validation is used for model selection and threshold calibration. The held-out test split provides a less biased final evaluation because it is not used to tune that threshold.

**Q: Are video and audio detection complete?**  
A: They are functional experimental paths, not fully validated detectors. Video averages sampled-frame scores from the image model and cannot detect temporal manipulation. Audio uses an external pretrained English speech classifier whose weights are downloaded on first use. DEEP-GUARD has not measured dedicated held-out video/audio benchmarks, and multimodal fusion is not yet validated.

**Q: What makes the project reproducible?**  
A: The repository publishes training/evaluation scripts, a notebook, checkpoint release reference, dataset manifest, checksums, and a machine-readable and human-readable evaluation report. A full clean-runtime reproduction is still a remaining verification task.

**Q: What would you improve next?**  
A: Train with more diverse real and AI-generated data, measure performance on separate held-out samples and unseen generators, improve uncertainty handling, and only then compare and publish the new checkpoint's metrics.

## 10. Links for the examiner

- [Source repository](https://github.com/ABHILATTHE6/DEEP-GUARD)
- [Final demonstration guide](FINAL_DEMO.md)
- [Evaluation report](../reports/image_evaluation_tiny_genimage.md)
- [Dataset manifest](../reports/dataset_manifest.json)
- [Training checkpoint release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints)
- [GitHub Actions checks](https://github.com/ABHILATTHE6/DEEP-GUARD/actions)
