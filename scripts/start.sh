#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ec2-user/axsy_inference"
APP_NAME="axsy-inference"
ECOSYSTEM_FILE="$ROOT_DIR/ecosystem.config.js"
SA_FILE="$ROOT_DIR/smart-vision-trainiing-sa.json"

cd "$ROOT_DIR"

if [[ -f "$SA_FILE" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$SA_FILE"
fi

if ! command -v pm2 >/dev/null 2>&1; then
  echo "pm2 not found. Install with: npm i -g pm2"
  exit 1
fi

if pm2 describe "$APP_NAME" >/dev/null 2>&1; then
  echo "Reloading PM2 app '$APP_NAME'..."
  pm2 reload "$ECOSYSTEM_FILE" --only "$APP_NAME" --update-env | cat
else
  echo "Starting PM2 app '$APP_NAME'..."
  pm2 start "$ECOSYSTEM_FILE" --only "$APP_NAME" --update-env | cat
fi

pm2 save | cat
echo "PM2 managed app '$APP_NAME' is running."



