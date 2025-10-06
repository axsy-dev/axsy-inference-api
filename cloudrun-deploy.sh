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
#   MEMORY (default: 8Gi)
#   CONCURRENCY (default: 1)
#   MIN_INSTANCES (default: 0)
#   MAX_INSTANCES (optional)
#   EXECUTION_ENV (default: gen2)
#   CPU_THROTTLING (default: false) - when false, CPU is always allocated
#   ALLOW_UNAUTH (default: true)
#   REQUIRE_MODELS (default: true) - require axsy-yolo.pt and axsy-classifier.pt to exist
#   REQUIRE_SA_JSON (default: true) - require smart-vision-trainiing-sa.json to exist

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
MEMORY=${MEMORY:-8Gi}
CONCURRENCY=${CONCURRENCY:-1}
MIN_INSTANCES=${MIN_INSTANCES:-1}
MAX_INSTANCES=${MAX_INSTANCES:-}
EXECUTION_ENV=${EXECUTION_ENV:-gen2}
CPU_THROTTLING=${CPU_THROTTLING:-false}
ALLOW_UNAUTH=${ALLOW_UNAUTH:-true}
REQUIRE_MODELS=${REQUIRE_MODELS:-true}
REQUIRE_SA_JSON=${REQUIRE_SA_JSON:-true}
SA_JSON="smart-vision-trainiing-sa.json"

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

# Ensure service account JSON is present so it is baked into the image and available at /app
if [[ "${REQUIRE_SA_JSON}" == "true" ]]; then
  if [[ ! -f "$SA_JSON" ]]; then
    echo "ERROR: Missing required service account JSON '$SA_JSON' in project root $(pwd)." >&2
    echo "Place $SA_JSON in the project root so Cloud Build includes it, or disable REQUIRE_SA_JSON." >&2
    exit 1
  fi
  echo "Found service account JSON: $SA_JSON"
fi

# Ensure repo exists
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location="$REGION" --description="Docker repo" --project="$PROJECT_ID"
fi

# Build and push image using Cloud Build
printf "\n==> Building and pushing image %s via Cloud Build\n" "$IMAGE_URI"
# Use a custom ignore file so .gitignore does not exclude the SA JSON from the build context
TMP_GCLOUDIGNORE=$(mktemp)
cat >"$TMP_GCLOUDIGNORE" <<'EOF'
# Custom Cloud Build ignore file
# Keep things small but DO NOT ignore JSON so service account is included
.git
__pycache__/
*.pyc
*.pyo
*.pyd
node_modules/
EOF
gcloud builds submit --ignore-file "$TMP_GCLOUDIGNORE" --tag "$IMAGE_URI" --project "$PROJECT_ID"
rm -f "$TMP_GCLOUDIGNORE"

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
  --execution-environment="$EXECUTION_ENV"
  --min-instances="$MIN_INSTANCES"
  --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/app/smart-vision-trainiing-sa.json,UVICORN_WORKERS=${UVICORN_WORKERS},UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL}"
  --timeout=600
)

if [[ "$ALLOW_UNAUTH" == "true" ]]; then
  ARGS+=(--allow-unauthenticated)
else
  ARGS+=(--no-allow-unauthenticated)
fi

# Optional max instances
if [[ -n "${MAX_INSTANCES}" ]]; then
  ARGS+=(--max-instances="$MAX_INSTANCES")
fi

# CPU throttling (false => always allocated CPU)
if [[ "$CPU_THROTTLING" == "false" ]]; then
  ARGS+=(--no-cpu-throttling)
else
  ARGS+=(--cpu-throttling)
fi

gcloud run deploy "$SERVICE" "${ARGS[@]}"

echo "\nDeployed. Service URL:"
gcloud run services describe "$SERVICE" --platform managed --region "$REGION" --project "$PROJECT_ID" --format='value(status.url)'
