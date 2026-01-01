#!/bin/bash
PID_DIR="./$(hostname)"
PID_FILE="${PID_DIR}/gpu-dryrun.pid"
LOG_FILE="${PID_DIR}/gpu-dryrun.log"
GPU_COUNT=$(echo $2 | tr ',' '\n' | wc -l)

case "$1" in
  up)
    if [ ! -d "$PID_DIR" ]; then
      mkdir -p "$PID_DIR"
    fi

    if [ -f "$PID_FILE" ]; then
      echo "Process is already running with PID $(cat $PID_FILE)"
      exit 1
    fi
    CUDA_VISIBLE_DEVICES=$2 nohup python gpu-dryrun.py > $LOG_FILE 2>&1 &
    MAIN_PID=$!
    echo $MAIN_PID > "$PID_FILE"
    # Capture child PIDs if any
    while true; do
      CHILD_PIDS=$(pgrep -P $MAIN_PID)
      CHILD_COUNT=$(echo "$CHILD_PIDS" | wc -w)
      if [ "$CHILD_COUNT" -eq $((GPU_COUNT + 1)) ]; then
          for PID in $CHILD_PIDS; do
              echo $PID >> "$PID_FILE"
          done
          break
      fi
      sleep 1
    done
    echo "Started gpu-dryrun.py with PIDs: $MAIN_PID $CHILD_PIDS"
    ;;
  down)
    if [ -f "$PID_FILE" ]; then
      while read -r PID; do
        if [ -n "$PID" ]; then
          kill "$PID" && echo "Stopped process with PID $PID"
        fi
      done < "$PID_FILE"
      rm -f "${PID_FILE}"
      rm -f "${LOG_FILE}"
      sleep 1
      rm -rf "${PID_DIR}"
    else
      echo "No processes are running."
    fi
    ;;
  *)
    echo "Usage: $0 {up <GPU_IDs>|down}"
    ;;
esac
