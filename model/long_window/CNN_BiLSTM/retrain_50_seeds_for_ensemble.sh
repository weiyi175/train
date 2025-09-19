#!/usr/bin/env bash
set -euo pipefail
# Retrain 50 seeds with final weights saving for ensemble averaging
# Usage: ./retrain_50_seeds_for_ensemble.sh

SEEDS=$(seq 1 50)
PARALLEL=${PARALLEL:-4}
EPOCHS=${EPOCHS:-60}
BASE_RESULT=${BASE_RESULT:-result_ensemble_weights}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

# Try to activate project virtualenv
if [ -z "${VIRTUAL_ENV:-}" ]; then
  REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
  if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate" || echo "[WARN] could not activate $REPO_ROOT/.venv"
  else
    echo "[INFO] .venv not found at $REPO_ROOT/.venv, proceeding with system python"
  fi
fi

command -v python >/dev/null 2>&1 || { echo "[FATAL] python not found in PATH after venv activation attempt"; exit 1; }

python -c "import os; os.makedirs('${BASE_RESULT}', exist_ok=True)" || true

echo "Retraining 50 seeds for ensemble weights..."
echo "Parallel: $PARALLEL  Epochs: $EPOCHS  Result base: $BASE_RESULT"

declare -a queue=()
launch() {
  local seed=$1
  local outdir=${BASE_RESULT}/seed${seed}
  mkdir -p "$outdir"
  echo "[launch] seed=$seed -> $outdir"

  nohup python train_cnn_bilstm.py \
    --checkpoint_metric precision_aware \
    --epochs "$EPOCHS" \
    --batch 64 \
    --result_dir "$outdir" \
    --run_seed "$seed" \
    --focal_alpha 0.5 \
    --class_weight_pos 1.0 \
    --mask_threshold 0.7 \
    --windows /home/user/projects/train/train_data/slipce/windows_npz.npz \
    > "$outdir/train.log" 2>&1 &
}

# Launch jobs with parallel limit
for seed in $SEEDS; do
  launch "$seed"
  queue+=($!)

  if [ ${#queue[@]} -ge $PARALLEL ]; then
    wait "${queue[0]}"
    queue=("${queue[@]:1}")
  fi
done

# Wait for remaining jobs
for pid in "${queue[@]}"; do
  wait "$pid"
done

echo "All retraining jobs completed!"

# Verify weights were saved
echo "Checking saved weights..."
for seed in $SEEDS; do
  weights_file="${BASE_RESULT}/seed${seed}/01/final.weights"
  if [ -f "$weights_file" ]; then
    echo "✓ seed$seed: weights saved"
  else
    echo "✗ seed$seed: weights missing"
  fi
done

echo "Ensemble weights preparation complete!"
