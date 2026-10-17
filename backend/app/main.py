"""DEEP-Guard FastAPI inference service.

The API supports image inference, experimental sampled-frame video screening, and a pretrained speech anti-spoofing baseline.
Each modality carries explicit limits in its result; video/audio are not DEEP-GUARD-trained or locally benchmarked models.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.analyzer import InvalidMediaFileError, UnsupportedMediaTypeError, analyze_file

app = FastAPI(title="DEEP-Guard API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = int(os.getenv("DEEP_GUARD_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))


def _media_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "deep-guard-api", "version": app.version}


@app.get("/api/v1/models")
def models() -> dict:
    return {
        "image": {"name": "deepguard-image-v1", "status": "ready_if_checkpoint_configured"},
        "video": {"name": "deepguard-image-efficientnet-b0-frame-aggregate", "status": "experimental_frame_based"},
        "audio": {"name": "SpeechAntiSpoofingBenchmarks/Wav2Vec2-Small-AntiDeepfake", "status": "experimental_pretrained", "license": "CC-BY-NC-SA-4.0"},
        "fusion": {"name": "deepguard-fusion-v1", "status": "planned_not_validated"},
    }


@app.post("/api/v1/analyze")
async def analyze(media: UploadFile = File(...)) -> dict:
    data = await media.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds upload limit")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = Path(media.filename or "upload.bin").suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        try:
            result = analyze_file(path, media.content_type or "application/octet-stream")
        except UnsupportedMediaTypeError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except InvalidMediaFileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail="A media-analysis dependency or model is unavailable. Check the ML dependencies and retry.",
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded media file is not valid or could not be read.",
            ) from exc
        result["media_id"] = _media_id(data)
        result["filename"] = media.filename or "upload"
        return result
    finally:
        Path(path).unlink(missing_ok=True)


# In container deployments, serve the production Vite bundle from the same origin
# as the API. In local development the Vite dev server continues to serve the UI.
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
FRONTEND_ASSETS = FRONTEND_DIST / "assets"

if FRONTEND_INDEX.is_file():
    if FRONTEND_ASSETS.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(FRONTEND_ASSETS)),
            name="frontend-assets",
        )

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_INDEX)
