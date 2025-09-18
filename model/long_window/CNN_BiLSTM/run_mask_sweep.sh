#!/usr/bin/env bash
set -euo pipefail

# Parameters via env with defaults
: "${EPOCHS:=50}"
: "${BATCH:=16}"
: "${ACC:=8}"
: "${THRESH_LIST:=0.6 0.75 0.8 0.85}"
: "${GATE:=}"   # e.g., 0.65; empty to disable
: "${WINDOWS:=/home/user/projects/train/train_data/slipce/windows_npz.npz}"
: "${RESULT_BASE:=result_mask_sweep}"

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
RESULT_BASE="$ROOT_DIR/${RESULT_BASE}"
mkdir -p "$RESULT_BASE"

# Ensure CUDA/WSL libs when running via nohup/non-interactive
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/wsl/lib:/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/extras/CUPTI/lib64"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib" ]; then
  export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
fi

echo "Starting mask sweep: epochs=$EPOCHS batch=$BATCH accum=$ACC thresholds=[$THRESH_LIST] gate=${GATE:-none}"

RUN_IDX=0
for TH in $THRESH_LIST; do
  RUN_IDX=$((RUN_IDX+1))
  OUT_DIR="$RESULT_BASE/run_$(printf '%02d' "$RUN_IDX")"
  mkdir -p "$OUT_DIR"
  echo "# $(date) : THRESH=$TH -> $OUT_DIR" | tee -a "$RESULT_BASE/sweep.log"
  # Build command
  CMD=(python3 "$ROOT_DIR/train_cnn_bilstm.py" \
       --windows "$WINDOWS" \
       --epochs "$EPOCHS" \
       --batch "$BATCH" \
       --accumulate_steps "$ACC" \
       --result_dir "$OUT_DIR" \
       --mask_threshold "$TH" \
       --mask_mode soft)
  if [ -n "$GATE" ]; then
    CMD+=(--window_mask_min_mean "$GATE")
  fi
  echo "+ ${CMD[*]}" | tee -a "$RESULT_BASE/sweep.log"
  "${CMD[@]}" >"$OUT_DIR/run.log" 2>&1 || echo "Run failed: TH=$TH (see $OUT_DIR/run.log)" | tee -a "$RESULT_BASE/sweep.log"
done

echo "Sweep done. See $RESULT_BASE and run_*/run.log"
