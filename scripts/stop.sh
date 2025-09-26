#!/usr/bin/env bash
set -euo pipefail

APP_NAME="axsy-inference"

if command -v pm2 >/dev/null 2>&1; then
  if pm2 describe "$APP_NAME" >/dev/null 2>&1; then
    pm2 stop "$APP_NAME" || true
    pm2 delete "$APP_NAME" || true
    echo "Stopped and deleted PM2 app '$APP_NAME'."
  else
    echo "PM2 app '$APP_NAME' not found."
  fi
else
  pkill -f "uvicorn server:get_app" || true
  echo "pm2 not installed. Killed matching uvicorn processes if any."
fi



