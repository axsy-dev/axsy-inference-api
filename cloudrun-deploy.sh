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
#   CPU (default: 1)
#   MEMORY (default: 1Gi)
#   CONCURRENCY (default: 80)
#   ALLOW_UNAUTH (default: true)

PROJECT_ID=${1:-}
REGION=${2:-}
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
CPU=${CPU:-1}
MEMORY=${MEMORY:-1Gi}
CONCURRENCY=${CONCURRENCY:-80}
ALLOW_UNAUTH=${ALLOW_UNAUTH:-true}

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
  --set-env-vars "UVICORN_WORKERS=${UVICORN_WORKERS},UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL}"
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
