#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/ec2-user/axsy_inference"
PID_FILE="$ROOT_DIR/uvicorn.pid"
LOG_FILE="$ROOT_DIR/uvicorn.log"
SA_FILE="$ROOT_DIR/smart-vision-trainiing-sa.json"

cd "$ROOT_DIR"

if [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE" 2>/dev/null)" > /dev/null 2>&1; then
  echo "Uvicorn already running with PID $(cat "$PID_FILE")."
  exit 0
fi

if [[ -f "$SA_FILE" ]]; then
  export GOOGLE_APPLICATION_CREDENTIALS="$SA_FILE"
fi

nohup uvicorn server:get_app \
  --factory \
  --host 0.0.0.0 \
  --port 3000 \
  --workers 1 \
  --log-level info \
  >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started Uvicorn (PID $(cat "$PID_FILE")) on 0.0.0.0:3000"



