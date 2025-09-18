#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export EPOCHS=${EPOCHS:-50}
export BATCH=${BATCH:-16}
export ACC=${ACC:-8}
export WINDOWS=${WINDOWS:-/home/user/projects/train/train_data/slipce/windows_npz.npz}
export MASK_THRESHOLD=${MASK_THRESHOLD:-0.6}
export RESULT_BASE=${RESULT_BASE:-result_focal_sweep}
export CLASS_WEIGHTS=${CLASS_WEIGHTS:-"1.0:1.0 1.0:0.8 1.0:0.6 1.2:1.0 0.8:1.2"}
export FOCAL_GRID=${FOCAL_GRID:-"0.25,1.5 0.25,2.0 0.2,1.5 0.2,2.0 0.1,1.0 0.1,1.5 0.4,1.0 0.4,1.5"}

mkdir -p "$RESULT_BASE"
echo "Launching focal sweep: cw x fg = $(echo $CLASS_WEIGHTS | wc -w) x $(echo $FOCAL_GRID | wc -w)"
nohup ./run_focal_sweep.sh > "$RESULT_BASE/sweep.nohup.out" 2>&1 &
pid=$!
echo $pid > "$RESULT_BASE/sweep.pid"
echo "Started PID=$pid. Logs: $RESULT_BASE/sweep.nohup.out"
