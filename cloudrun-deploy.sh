#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./cloudrun-deploy.sh <gcp_project_id> <region> <service_name>
# Env vars used if present:
#   REPO (Artifact Registry repo, default: default repo named 'containers')
#   IMAGE_TAG (default: latest)
#   PORT (default: 8080)
#   UVICORN_WORKERS (default: 1)
#   UVICORN_LOG_LEVEL (default: info)
#   CPU (default: 2)
#   MEMORY (default: 2Gi)
#   CONCURRENCY (default: 80)
#   ALLOW_UNAUTH (default: true)
#   REQUIRE_MODELS (default: true) - require axsy-yolo.pt and axsy-classifier.pt to exist

PROJECT_ID=${1:-smart-vision-training}
REGION=${2:-europe-west1}
SERVICE=${3:-axsy-inference}

if [[ -z "${PROJECT_ID}" || -z "${REGION}" ]]; then
  echo "Usage: $0 <gcp_project_id> <region> <service_name>" >&2
  exit 1
fi

REPO=${REPO:-containers}
IMAGE_TAG=${IMAGE_TAG:-latest}
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:${IMAGE_TAG}"

PORT=${PORT:-8080}
UVICORN_WORKERS=${UVICORN_WORKERS:-1}
UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL:-info}
CPU=${CPU:-2}
MEMORY=${MEMORY:-2Gi}
CONCURRENCY=${CONCURRENCY:-80}
ALLOW_UNAUTH=${ALLOW_UNAUTH:-true}
REQUIRE_MODELS=${REQUIRE_MODELS:-true}

# Preflight: ensure default model files exist so the container has them at /app
if [[ "${REQUIRE_MODELS}" == "true" ]]; then
  missing=0
  for f in axsy-yolo.pt axsy-classifier.pt; do
    if [[ ! -f "$f" ]]; then
      echo "ERROR: Missing required model file '$f' in project root $(pwd)." >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Place both axsy-yolo.pt and axsy-classifier.pt in the project root, then re-run." >&2
    exit 1
  fi
  echo "Found default models:"; ls -lh axsy-yolo.pt axsy-classifier.pt
fi

# Ensure repo exists
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --description="Docker repo" --project="$PROJECT_ID"
fi

# Build and push image using Cloud Build
printf "\n==> Building and pushing image %s via Cloud Build\n" "$IMAGE_URI"
gcloud builds submit --tag "$IMAGE_URI" --project "$PROJECT_ID"

# Deploy to Cloud Run
printf "\n==> Deploying Cloud Run service %s in %s\n" "$SERVICE" "$REGION"
ARGS=(
  --project="$PROJECT_ID"
  --region="$REGION"
  --image="$IMAGE_URI"
  --platform=managed
  --port="$PORT"
  --cpu="$CPU"
  --memory="$MEMORY"
  --concurrency="$CONCURRENCY"
  --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/app/smart-vision-training-sa.json,UVICORN_WORKERS=${UVICORN_WORKERS},UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL}"
  --timeout=600
)

if [[ "$ALLOW_UNAUTH" == "true" ]]; then
  ARGS+=(--allow-unauthenticated)
else
  ARGS+=(--no-allow-unauthenticated)
fi

gcloud run deploy "$SERVICE" "${ARGS[@]}"

echo "\nDeployed. Service URL:"
gcloud run services describe "$SERVICE" --platform managed --region "$REGION" --project "$PROJECT_ID" --format='value(status.url)'
