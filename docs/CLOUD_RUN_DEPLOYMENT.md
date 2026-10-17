# Public Demo Deployment (Google Cloud Run)

This deploys the existing React/Vite dashboard and FastAPI image, experimental video-frame, and external pretrained audio inference routes together as one public URL. It uses the current published checkpoint; it does not retrain the model or alter the evaluation report.

## Hosting and cost notes

Cloud Run supports deploying from source and uses the repository's `Dockerfile` when one is present. The command below sets one CPU, 2 GiB RAM, scale-to-zero, and at most one instance. That is a conservative demo configuration, not a promise that hosting will always cost zero.

Google Cloud requires a project and billing setup for Cloud Run deployment. Cloud Run has an always-free usage tier, but usage beyond the free tier and other services such as container-image storage or build usage can incur charges. Review the current [Cloud Run pricing](https://cloud.google.com/run/pricing), set a billing budget alert, and confirm you are comfortable enabling billing before continuing. Budget alerts notify you; they are not a hard spending cap.

This creates a public, unauthenticated image-analysis endpoint. For the demo, upload only non-sensitive images. Do not use it for private photos or production/high-stakes decisions.

## Prerequisites

- A Google Cloud account.
- A Google Cloud project with billing enabled and permission to deploy Cloud Run services.
- Google Cloud CLI (`gcloud`) installed.
- A local checkout of this repository on the current `main` branch.

## Deploy from Windows PowerShell

From the repository root:

```powershell
git pull --ff-only origin main
gcloud auth login
gcloud projects list
gcloud config set project YOUR_GOOGLE_CLOUD_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy deep-guard --source . --region asia-south1 --allow-unauthenticated --memory 2Gi --cpu 1 --concurrency 1 --max 1 --min 0 --timeout 180
```

Replace `YOUR_GOOGLE_CLOUD_PROJECT_ID` with your actual project ID. Do not paste billing information, credentials, or access tokens into this repository.

The first build can take several minutes because it builds the frontend, installs CPU-only PyTorch and audio/video dependencies, and downloads the published image checkpoint. The first audio analysis after deployment downloads approximately 1.26 GB of separate speech-model weights, which can add startup latency and storage/network use. The Dockerfile checks the checkpoint SHA-256 during the build and fails if the downloaded bytes don't match the currently published baseline.

When deployment completes, copy the HTTPS service URL printed by `gcloud`.

## Verify the public service

Open these URLs, replacing `YOUR_SERVICE_URL` with the deployed URL:

- `https://YOUR_SERVICE_URL/` — dashboard.
- `https://YOUR_SERVICE_URL/api/v1/health` — API health JSON, expected to include `"status":"ok"`.
- `https://YOUR_SERVICE_URL/api/v1/models` — modality-status JSON.

Then upload a small non-sensitive image and run detection. The first analysis after a cold start may take longer. A valid public deployment does not mean the classifier will correctly label every image; keep the published benchmark and limitations visible.

## Stop hosting when finished

If you only need the demo during submission, you can delete the Cloud Run service after your presentation from Google Cloud Console, or run:

```powershell
gcloud run services delete deep-guard --region asia-south1
```

Review the Cloud Run and Artifact Registry pages in the Google Cloud Console after deletion to confirm no other deployed resources or stored images remain. Artifact storage or build resources may have separate charges.

## Troubleshooting

- **Build fails while fetching the checkpoint:** confirm the public `training-checkpoints` release still includes `image-training-best.pt`. The build checks its SHA-256 against the current baseline value.
- **Container fails to start:** inspect the Cloud Run revision logs; confirm the service is using the committed `Dockerfile`.
- **Dashboard loads but API is offline:** this deployment serves the built frontend and API from the same origin. Confirm the latest source was pulled and redeployed.
- **Memory issue:** the deployment uses 2 GiB. Check logs before changing memory or concurrency.
- **Unexpected cost concern:** stop/delete the service and review project billing. The free tier is limited and not a guaranteed cap.

## Related project artifacts

- [Final demonstration guide](FINAL_DEMO.md)
- [Presentation and viva guide](PRESENTATION_AND_VIVA.md)
- [Held-out image evaluation report](../reports/image_evaluation_tiny_genimage.md)
- [Dataset manifest](../reports/dataset_manifest.json)
- [Checkpoint release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints)
