#!/usr/bin/env bash
set -euo pipefail
# Run multiple seeds with precision-focused adjustments.
# Adjustments: focal_alpha=0.5, class_weight_pos=1.0, mask_threshold=0.7
# Usage: SEEDS="1 2 ..." CHECK_METRIC=precision_aware ./run_multi_seed_precision_focus.sh
# Env vars:
#   SEEDS: space-separated list (default 1..20)
#   CHECK_METRIC: auc|composite|precision_aware (default precision_aware)
#   PARALLEL: number of concurrent processes (default 2)
#   EPOCHS: override epochs (default 70)
#   BASE_RESULT: result dir (default result_multi_seed_precision_focus)
#   EXTRA_ARGS: extra CLI args appended (default includes precision adjustments)

SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"}
CHECK_METRIC=${CHECK_METRIC:-precision_aware}
PARALLEL=${PARALLEL:-2}
EPOCHS=${EPOCHS:-70}
BASE_RESULT=${BASE_RESULT:-result_multi_seed_precision_focus}
# Default precision-focused args
DEFAULT_EXTRA="--focal_alpha 0.5 --class_weight_pos 1.0 --mask_threshold 0.7"
EXTRA_ARGS=${EXTRA_ARGS:-"$DEFAULT_EXTRA"}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"
# Try to activate project virtualenv (expected at repo root: train/.venv)
if [ -z "${VIRTUAL_ENV:-}" ]; then
  # ascend three levels to reach repository root (CNN_BiLSTM -> long_window -> model -> train)
  REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
  if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate" || echo "[WARN] could not activate $REPO_ROOT/.venv"
  else
    echo "[INFO] .venv not found at $REPO_ROOT/.venv, proceeding with system python"
  fi
fi

command -v python >/dev/null 2>&1 || { echo "[FATAL] python not found in PATH after venv activation attempt"; exit 1; }

python -c "import os; os.makedirs('${BASE_RESULT}', exist_ok=True)" || true

echo "Running seeds: $SEEDS" 
echo "Checkpoint metric: $CHECK_METRIC  Parallel: $PARALLEL  Result base: $BASE_RESULT"

declare -a queue=()
launch() {
  local seed=$1
  local outdir=${BASE_RESULT}/seed${seed}
  mkdir -p "$outdir"
  echo "[launch] seed=$seed -> $outdir"
  nohup python train_cnn_bilstm.py \
    --checkpoint_metric "$CHECK_METRIC" \
    --run_seed "$seed" \
    --result_dir "$outdir" \
    --epochs "$EPOCHS" \
    $EXTRA_ARGS \
    > "$outdir/train.nohup.out" 2>&1 &
  echo $! > "$outdir/pid"
}

for s in $SEEDS; do
  launch "$s"
  queue+=("$s")
  while [ ${#queue[@]} -ge $PARALLEL ]; do
    sleep 5
    alive=()
    for sd in "${queue[@]}"; do
      pidfile=${BASE_RESULT}/seed${sd}/pid
      if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile" || true)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
          alive+=("$sd")
        else
          echo "[done] seed $sd finished"
        fi
      fi
    done
    queue=("${alive[@]}")
  done
done

echo "All seeds launched. Waiting for remaining: ${queue[*]}"
while [ ${#queue[@]} -gt 0 ]; do
  sleep 10
  alive=()
  for sd in "${queue[@]}"; do
    pidfile=${BASE_RESULT}/seed${sd}/pid
    pid=$(cat "$pidfile" 2>/dev/null || true)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      alive+=("$sd")
    else
      echo "[done] seed $sd finished"
    fi
  done
  queue=("${alive[@]}")
  echo "Remaining: ${queue[*]}"
done

echo "Multi-seed run complete: $BASE_RESULT"
