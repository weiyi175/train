#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <seed> [result_dir]"
  exit 2
fi
SEED=$1
RESULT_DIR=${2:-result_confirm_seed${SEED}}
cd "$(dirname "$0")"
source /home/user/projects/train/.venv/bin/activate
mkdir -p "$RESULT_DIR"
nohup python3 train_cnn_bilstm.py --result_dir "$RESULT_DIR" --run_seed "$SEED" --no_early_stop > "$RESULT_DIR/sweep.nohup.out" 2>&1 &
PID=$!
printf "%s" "$PID" > "$RESULT_DIR/sweep.pid"
echo "Started seed=$SEED PID=$PID -> $RESULT_DIR"
