#!/bin/bash
PID_FILE="gpu-dryrun.pid"

case "$1" in
  up)
    if [ -f "$PID_FILE" ]; then
      echo "Process is already running with PID $(cat $PID_FILE)"
      exit 1
    fi
    CUDA_VISIBLE_DEVICES=$2 nohup python gpu-dryrun.py > gpu-dryrun.log 2>&1 &
    MAIN_PID=$!
    echo $MAIN_PID > "$PID_FILE"
    # Capture child PIDs if any
    sleep 10
    CHILD_PIDS=$(pgrep -P $MAIN_PID)
    if [ -n "$CHILD_PIDS" ]; then
        for PID in $CHILD_PIDS; do
            echo $PID >> "$PID_FILE"
        done
    fi
    echo "Started gpu-dryrun.py with PIDs: $MAIN_PID $CHILD_PIDS"
    ;;
  down)
    if [ -f "$PID_FILE" ]; then
      while read -r PID; do
        if [ -n "$PID" ]; then
          kill "$PID" && echo "Stopped process with PID $PID"
        fi
      done < "$PID_FILE"
      rm -f "$PID_FILE"
    else
      echo "No processes are running."
    fi
    ;;
  *)
    echo "Usage: $0 {up <GPU_IDs>|down}"
    ;;
esac