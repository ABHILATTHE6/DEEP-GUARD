# Video and Audio Phase — Implementation Notes

DEEP-GUARD now accepts image, video, and audio files through the same dashboard upload and FastAPI `/api/v1/analyze` endpoint. The three modalities do **not** have equal maturity; keep the labels and limitations below visible in the project presentation.

## Current modality status

| Modality | Runtime behavior | Status and limits |
|---|---|---|
| Image | Existing EfficientNet-B0 checkpoint; same threshold and reported evaluation | Project-trained baseline, held-out Tiny-GenImage results published |
| Video | Samples up to 8 frames, applies the image checkpoint in a batch, aggregates mean AI/real class scores | Experimental frame screen; no temporal model, no audio/video synchronization or lip-sync analysis, no dedicated held-out video benchmark |
| Audio | Loads mono speech at 16 kHz and runs an external pretrained Wav2Vec2 speech detector | Experimental external model; weights download on first audio request (about 1.26 GB), English-focused, no DEEP-GUARD-specific audio benchmark |
| Multimodal fusion | No combined image/video/audio decision is produced | Not implemented/validated end to end |

## Install local serving dependencies

On Windows PowerShell from the repository root, with the virtual environment active:

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -e ".[serving,dev]"
cd frontend
npm ci
npm run build
```

The serving extra installs PyTorch/torchvision, Transformers, headless OpenCV, librosa, soundfile, and Pillow. The first real audio analysis requires an internet connection to download the upstream audio model to the local Hugging Face cache. A first audio request may be slow and requires sufficient disk space and RAM. Image and frame-aggregated video inference do not require downloading that audio model.

## Run the multimodal demo

Start the backend with the existing image checkpoint configured, then start the frontend as described in [Local Demo Quick Start](LOCAL_DEMO_QUICKSTART.md). In the dashboard, select an image, MP4/WebM video, or supported speech-audio file and click **Run detection**.

For video, the dashboard shows average frame-based scores and sampled-frame evidence. The classifier's image threshold has not been calibrated for video; the aggregate uses a simple experimental 0.5 decision boundary and a small uncertainty band.

For audio, the model card is [garystafford/wav2vec2-deepfake-voice-detector](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector). The upstream model uses Apache-2.0 licensing and documents output class 0 as real and class 1 as fake. It was fine-tuned on 1,866 short clips spanning real YouTube speech and six TTS sources; these model-card results are not DEEP-GUARD test results. Its authors caution that additional cross-engine testing is needed. Do not report those external metrics as if we measured them.

## Validation in CI

The container CI job builds the frontend and API container, starts the service, checks health and the dashboard page, sends a generated image through image inference, sends a generated video through frame aggregation, and tests the audio preprocessing/model-output contract with an injected test model. This tests our code paths without downloading the 1.26 GB upstream audio weights on every CI run; the external weight-download and inference path still requires first-run validation on the user's machine.

## What would make video/audio fully complete

1. Train or adopt an appropriate dedicated video detector and benchmark it on a held-out video dataset. Include manipulation types, unseen generators, video codecs/resolutions, and sample-level metrics. Frame aggregation alone is not a dedicated video detector.
2. Build a licensed audio train/validation/test benchmark with voices and generators kept disjoint across splits. Validate the external model on a permissible out-of-domain dataset or train a DEEP-GUARD-specific model; do not evaluate on any corpora prohibited by its model card.
3. Calibrate each modality's threshold on its own validation data and publish confusion matrices, precision/recall, F1, ROC-AUC, and domain limitations.
4. Only then implement a fusion model that is calibrated and evaluated end to end. Until then the app reports each modality separately.

## Existing image baseline remains unchanged

The current image checkpoint, Tiny-GenImage reports, threshold, and public release asset are not overwritten by these additions. See the [image evaluation report](../reports/image_evaluation_tiny_genimage.md) for that separate baseline.