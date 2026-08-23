"""DEEP-Guard FastAPI inference service.

The API exposes a stable contract for image, video, audio and multimodal analysis.
Image inference uses the trained EfficientNet pipeline when a checkpoint is supplied;
video/audio adapters return an explicit unavailable status until their models are trained.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.services.analyzer import analyze_file

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
        "video": {"name": "deepguard-video-v1", "status": "experimental"},
        "audio": {"name": "deepguard-audio-v1", "status": "experimental"},
        "fusion": {"name": "deepguard-fusion-v1", "status": "score_fusion_contract"},
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
        result = analyze_file(path, media.content_type or "application/octet-stream")
        result["media_id"] = _media_id(data)
        result["filename"] = media.filename or "upload"
        return result
    finally:
        Path(path).unlink(missing_ok=True)
