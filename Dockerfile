# Multi-stage Dockerfile for Axsy Inference API on Cloud Run

# Stage 1: builder for dependencies (wheels)
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        python3-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# Stage 2: runtime image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    UVICORN_WORKERS=1 \
    UVICORN_LOG_LEVEL=info

# Cloud Run runs as non-root by default; create a user and dir
RUN useradd -m appuser
WORKDIR /app

# Install runtime dependencies needed by pillow/ultralytics (libgl etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt

COPY . .

# Ensure local model files (if any) are readable and a writable cache dir exists
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Start with uvicorn using the app factory; allow PORT/WORKERS to be overridden
CMD ["sh", "-c", "uvicorn server:get_app --factory --host 0.0.0.0 --port ${PORT:-8080} --workers ${UVICORN_WORKERS:-1} --log-level ${UVICORN_LOG_LEVEL:-info}"]


