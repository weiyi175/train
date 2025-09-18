#!/usr/bin/env bash
set -euo pipefail

# Launch the mask-threshold sweep in background via nohup with safe env passing.
# Env overrides supported: EPOCHS, BATCH, ACC, GATE, THRESH_LIST

cd "$(dirname "$0")"

export EPOCHS=${EPOCHS:-50}
export BATCH=${BATCH:-16}
export ACC=${ACC:-8}
export GATE=${GATE:-}
export THRESH_LIST=${THRESH_LIST:-"0.6 0.75 0.8 0.85"}
export RESULT_BASE=${RESULT_BASE:-result_mask_sweep}

mkdir -p "$RESULT_BASE"

echo "Launching sweep: epochs=$EPOCHS batch=$BATCH acc=$ACC gate=${GATE:-none} thresholds=[$THRESH_LIST]"
nohup ./run_mask_sweep.sh > "$RESULT_BASE/sweep.nohup.out" 2>&1 &
pid=$!
echo $pid > "$RESULT_BASE/sweep.pid"
echo "Started PID=$pid. Logs: $RESULT_BASE/sweep.nohup.out (pid file: $RESULT_BASE/sweep.pid)"
