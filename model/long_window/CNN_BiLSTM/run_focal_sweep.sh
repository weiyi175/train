#!/usr/bin/env bash
set -euo pipefail

# Grid:
# - CLASS_WEIGHTS: space-separated pairs like "1.0:1.0 1.0:0.8 1.0:0.6 1.2:1.0 0.8:1.2"
# - FOCAL_GRID: space-separated alpha,gamma tuples like "0.25,1.5 0.25,2.0 0.2,1.5 0.2,2.0 0.1,1.0 0.1,1.5 0.4,1.0 0.4,1.5"

: "${EPOCHS:=50}"
: "${BATCH:=16}"
: "${ACC:=8}"
: "${WINDOWS:=/home/user/projects/train/train_data/slipce/windows_npz.npz}"
: "${MASK_THRESHOLD:=0.6}"
: "${RESULT_BASE:=result_focal_sweep}"
: "${CLASS_WEIGHTS:=1.0:1.0 1.0:0.8 1.0:0.6 1.2:1.0 0.8:1.2}"
: "${FOCAL_GRID:=0.25,1.5 0.25,2.0 0.2,1.5 0.2,2.0 0.1,1.0 0.1,1.5 0.4,1.0 0.4,1.5}"

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
OUT_ROOT="$ROOT_DIR/${RESULT_BASE}"
mkdir -p "$OUT_ROOT"

echo "Focal sweep -> epochs=$EPOCHS batch=$BATCH acc=$ACC mask_threshold=$MASK_THRESHOLD"
echo "Class weights: $CLASS_WEIGHTS"
echo "Focal grid (alpha,gamma): $FOCAL_GRID"

cw_idx=0
for cw in $CLASS_WEIGHTS; do
  cw_idx=$((cw_idx+1))
  WN=${cw%%:*}
  WP=${cw##*:}
  fg_idx=0
  for ag in $FOCAL_GRID; do
    fg_idx=$((fg_idx+1))
    A=${ag%%,*}
    G=${ag##*,}
    RUN_DIR="$OUT_ROOT/cw$(printf '%02d' $cw_idx)_fg$(printf '%02d' $fg_idx)"
    mkdir -p "$RUN_DIR"
    echo "# $(date) -> class_weight=${WN}:${WP}, alpha=$A, gamma_end=$G -> $RUN_DIR" | tee -a "$OUT_ROOT/sweep.log"
    python3 "$ROOT_DIR/train_cnn_bilstm.py" \
      --windows "$WINDOWS" \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --accumulate_steps "$ACC" \
      --result_dir "$RUN_DIR" \
      --mask_threshold "$MASK_THRESHOLD" \
      --mask_mode soft \
      --focal_alpha "$A" \
      --focal_gamma_start 0.0 \
      --focal_gamma_end "$G" \
      --curriculum_epochs 20 \
      --class_weight_neg "$WN" \
      --class_weight_pos "$WP" \
      > "$RUN_DIR/run.log" 2>&1 || echo "Run failed at $RUN_DIR" | tee -a "$OUT_ROOT/sweep.log"
  done
done

echo "Focal sweep finished. Results in $OUT_ROOT"
