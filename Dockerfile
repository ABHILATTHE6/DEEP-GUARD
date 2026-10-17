# syntax=docker/dockerfile:1

FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    DEEP_GUARD_IMAGE_CHECKPOINT=/app/artifacts/image_training_best.pt

WORKDIR /app

COPY pyproject.toml ./
COPY backend/ ./backend/
COPY ml/ ./ml/

RUN python -m pip install --upgrade pip \
    && python -m pip install -e . \
    && python -m pip install "pillow>=10,<12" \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
       torch==2.5.1+cpu torchvision==0.20.1+cpu

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Fetch the public baseline checkpoint and verify the published SHA-256 before
# including it in the container image. Never put a GitHub token in this build.
RUN mkdir -p /app/artifacts \
    && python -c "from urllib.request import urlretrieve; urlretrieve('https://github.com/ABHILATTHE6/DEEP-GUARD/releases/download/training-checkpoints/image-training-best.pt', '/app/artifacts/image_training_best.pt')" \
    && echo "7f34d91a65b6215dcee4024cd2f65177d6687a23be0d59a10d67c1d4dab21e1b  /app/artifacts/image_training_best.pt" | sha256sum -c -

EXPOSE 8080
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
