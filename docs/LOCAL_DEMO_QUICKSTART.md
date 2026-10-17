# Local Demo Quick Start (Windows)

This is the simplest way to launch the working local DEEP-GUARD image demo. It does not require a cloud account, public deployment, or model retraining.

## One-command startup

First, make sure the saved checkpoint exists at `artifacts/image_training_best.pt`, the Python virtual environment is installed, and frontend dependencies are installed. From the repository root, run `.\\.venv\\Scripts\\python.exe -m pip install -e ".[serving,dev]"` once to install the image/video/audio serving dependencies. If needed, run `npm ci` once from `frontend`.

From the repository root in PowerShell:

```powershell
git pull --ff-only origin main
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

The script verifies the local virtual environment, saved checkpoint, and frontend dependencies. It then opens the API and frontend in separate PowerShell windows.

- Dashboard: <http://localhost:5173>
- API health: <http://127.0.0.1:8000/api/v1/health>

Leave both service windows running. Stop each service with `Ctrl+C` in its corresponding window.

The first audio analysis downloads about 1.26 GB of external pretrained model weights and requires an internet connection. Video analysis samples up to eight frames with the image model; it is experimental and does not inspect temporal artifacts.

## If startup does not work

- **Checkpoint missing:** follow [Image Training and Checkpoint Recovery](TRAINING.md) to download the published `image-training-best.pt` release asset into `artifacts/image_training_best.pt`.
- **Python environment missing:** create the project virtual environment and install the project dependencies using the repository setup instructions.
- **Frontend dependencies missing:** run `cd frontend` then `npm ci`, and return to the repository root.
- **Port already in use:** close the old backend/Vite terminal, or stop the service using port 8000 / 5173 before retrying.

## Project demo boundary

The local demo is the verified presentation path. It runs on your computer and is not publicly reachable by examiners without a separate deployment. The current operational detector is the image baseline; video/audio are experimental/untrained and fusion has not been validated end to end. Do not present individual predictions as proof of authenticity.

Published baseline: [evaluation report](../reports/image_evaluation_tiny_genimage.md), [dataset manifest](../reports/dataset_manifest.json), and [checkpoint release](https://github.com/ABHILATTHE6/DEEP-GUARD/releases/tag/training-checkpoints).