#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SWEEP_DIR="result_mask_sweep"
PLAN_THRESH=( ${THRESH_LIST:-0.6 0.75 0.8 0.85} )

echo "Watcher started. Waiting for results.json of ${#PLAN_THRESH[@]} runs..."

# Wait loop: for each run_XX, expect results.json under the first subdir (01)
deadline=$(( $(date +%s) + ${TIMEOUT_SEC:-86400} ))
for i in $(seq 1 ${#PLAN_THRESH[@]}); do
  run_dir=$(printf "%s/run_%02d/01" "$SWEEP_DIR" "$i")
  res="$run_dir/results.json"
  echo "Waiting for $res"
  while [ ! -f "$res" ]; do
    if [ $(date +%s) -gt $deadline ]; then
      echo "Timeout waiting for $res" >&2
      exit 1
    fi
    sleep 30
  done
  echo "Found: $res"
done

echo "All runs completed. Aggregating..."
python3 ./aggregate_mask_sweep.py
python3 ./post_sweep_report.py
echo "Aggregation done. See $SWEEP_DIR/aggregate.csv and $SWEEP_DIR/report.md"
