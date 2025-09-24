#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ec2-user/axsy_inference"
PID_FILE="$ROOT_DIR/uvicorn.pid"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" || true)
  if [[ -n "${PID:-}" ]] && ps -p "$PID" > /dev/null 2>&1; then
    kill "$PID" || true
    sleep 1
  fi
  rm -f "$PID_FILE"
  echo "Stopped Uvicorn."
else
  # Fallback: kill by pattern if no pid file
  pkill -f "uvicorn server:get_app" || true
  echo "No PID file. Killed matching uvicorn processes if any."
fi



