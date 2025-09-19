#!/usr/bin/env bash
# Monitor ensemble training progress and run evaluation when complete

WEIGHTS_DIR="result_ensemble_weights"
EXPECTED_SEEDS=50

echo "Monitoring ensemble training progress..."
echo "Expected: $EXPECTED_SEEDS seeds"

while true; do
    # Count completed seeds (those with results.json)
    completed=$(find "$WEIGHTS_DIR" -name "results.json" | wc -l)
    echo "Completed: $completed/$EXPECTED_SEEDS seeds"

    if [ "$completed" -ge "$EXPECTED_SEEDS" ]; then
        echo "All seeds completed! Running ensemble evaluation..."

        # Run top-k ensemble evaluation
        for k in 5 10 15 20; do
            echo "Running Top-$k ensemble..."
            python ensemble_topk_logits.py --weights_dir "$WEIGHTS_DIR" --k "$k" --metric precision_aware
        done

        echo "Ensemble evaluation complete!"
        break
    fi

    echo "Waiting 60 seconds before next check..."
    sleep 60
done
