#!/usr/bin/env python3
"""
Top-K Logit Averaging Ensemble for CNN-BiLSTM Models
Usage: python ensemble_topk_logits.py --weights_dir result_ensemble_weights --k 10
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import tensorflow as tf
from model import build_cnn_bilstm, mlp_attention
from utils import load_npz_windows

def load_model_with_weights(model, weights_path):
    """Load model weights from file"""
    try:
        model.load_weights(weights_path)
        return True
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")
        return False

def get_seed_performance(seed_dir):
    """Get performance metrics for a seed"""
    results_file = os.path.join(seed_dir, '01', 'results.json')
    if not os.path.exists(results_file):
        return None

    try:
        import json
        with open(results_file, 'r') as f:
            data = json.load(f)
        return {
            'seed': int(seed_dir.split('seed')[-1]),
            'auc': data['test']['auc'],
            'f1': data['test']['f1'],
            'recall': data['test']['recall'],
            'precision': data['test']['precision'],
            'precision_aware': data['test']['precision_aware'],
            'composite': data['test']['composite']
        }
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Top-K Logit Averaging Ensemble')
    parser.add_argument('--weights_dir', type=str, default='result_ensemble_weights',
                       help='Directory containing seed results with weights')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of top models to ensemble')
    parser.add_argument('--metric', type=str, default='precision_aware',
                       choices=['auc', 'f1', 'recall', 'precision', 'precision_aware', 'composite'],
                       help='Metric to rank models')
    parser.add_argument('--windows', type=str,
                       default='/home/user/projects/train/train_data/slipce/windows_npz.npz',
                       help='Path to windows NPZ file')
    args = parser.parse_args()

    print(f"Starting Top-{args.k} Logit Averaging Ensemble")
    print(f"Using metric: {args.metric}")
    print(f"Weights directory: {args.weights_dir}")

    # Load test data
    print("Loading test data...")
    try:
        X_test, y_test = load_npz_windows(args.windows)
        print(f"Test data loaded: {X_test.shape[0]} samples")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # For simplicity, assume no mask data in ensemble evaluation
    m_test = None

    # Collect all seed performances
    seed_dirs = [d for d in os.listdir(args.weights_dir)
                if d.startswith('seed') and os.path.isdir(os.path.join(args.weights_dir, d))]
    seed_dirs.sort(key=lambda x: int(x.replace('seed', '')))

    performances = []
    valid_seeds = []

    print(f"Found {len(seed_dirs)} seed directories")
    for seed_dir in seed_dirs:
        perf = get_seed_performance(os.path.join(args.weights_dir, seed_dir))
        if perf is not None:
            performances.append(perf)
            valid_seeds.append(seed_dir)

    if len(performances) == 0:
        print("No valid seed performances found!")
        return

    # Convert to DataFrame and rank
    df_perf = pd.DataFrame(performances)
    df_perf = df_perf.sort_values(args.metric, ascending=False)
    print(f"\nTop {min(args.k, len(df_perf))} seeds by {args.metric}:")
    print(df_perf.head(args.k)[['seed', args.metric, 'auc', 'f1', 'precision_aware']].to_string(index=False))

    # Select top-k seeds
    top_k_seeds = df_perf.head(args.k)['seed'].tolist()
    print(f"\nSelected top-{args.k} seeds: {top_k_seeds}")

    # Build model template
    print("\nBuilding model template...")
    try:
        # For long windows: (features=36, timesteps=75)
        # For short windows: (features=36, timesteps=30)
        if X_test.shape[1] == 36 and X_test.shape[2] == 75:
            # Long windows
            input_shape = (36, 75)
        elif X_test.shape[1] == 36 and X_test.shape[2] == 30:
            # Short windows
            input_shape = (36, 30)
        else:
            # Fallback: infer from data
            input_shape = (X_test.shape[1], X_test.shape[2])

        print(f"Using input shape: {input_shape}")

        model_template = build_cnn_bilstm(
            input_shape=input_shape,
            num_filters=64,
            kernel_sizes=(3, 5, 3),
            conv_dropout=0.2,
            lstm_units=64,  # Match training script
            lstm_dropout=0.2,
            attn_units=32,
            use_mask=False
        )
        print("Model template built successfully")
    except Exception as e:
        print(f"Error building model template: {e}")
        return

    # Collect predictions from top-k models
    all_logits = []
    successful_models = 0

    print(f"\nLoading weights and collecting predictions from top-{args.k} models...")
    for seed in top_k_seeds:
        seed_dir = f"seed{seed}"
        weights_path = os.path.join(args.weights_dir, seed_dir, '01', 'final.weights')

        # Check if weights file exists (TensorFlow checkpoint format)
        weights_index = weights_path + '.index'
        if not os.path.exists(weights_index):
            print(f"Warning: Weights index not found for seed {seed}: {weights_index}")
            continue

        try:
            # Create fresh model instance
            model = tf.keras.models.clone_model(model_template)
            model.compile(optimizer='adam', loss='binary_crossentropy')

            # Load weights
            if load_model_with_weights(model, weights_path):
                # Make predictions
                if m_test is not None:
                    logits = model.predict([X_test, m_test], verbose=0, batch_size=32)
                else:
                    logits = model.predict(X_test, verbose=0, batch_size=32)

                # Ensure logits are 1D probabilities
                if logits.ndim > 1 and logits.shape[-1] > 1:
                    logits = logits[:, 1]  # Take positive class
                elif logits.ndim > 1:
                    logits = logits.squeeze()

                all_logits.append(logits)
                successful_models += 1
                print(f"✓ Loaded seed {seed}")
            else:
                print(f"✗ Failed to load seed {seed}")

        except Exception as e:
            print(f"Error processing seed {seed}: {e}")
            continue

    if len(all_logits) == 0:
        print("No models could be loaded successfully!")
        return

    print(f"\nSuccessfully loaded {successful_models}/{args.k} models")

    # Average logits
    print("Averaging logits...")
    ensemble_logits = np.mean(all_logits, axis=0)
    ensemble_preds = (ensemble_logits >= 0.5).astype(int)

    # Calculate ensemble metrics
    ensemble_auc = roc_auc_score(y_test, ensemble_logits)
    ensemble_f1 = f1_score(y_test, ensemble_preds)
    ensemble_recall = recall_score(y_test, ensemble_preds)
    ensemble_precision = precision_score(y_test, ensemble_preds)

    tn, fp, fn, tp = confusion_matrix(y_test, ensemble_preds).ravel()
    ensemble_composite = 0.5 * ensemble_recall + 0.3 * ensemble_f1 + 0.2 * ensemble_auc
    ensemble_precision_aware = 0.5 * ensemble_precision + 0.3 * ensemble_f1 + 0.2 * ensemble_auc

    # Compare with individual best
    best_individual = df_perf.iloc[0]
    print("\n" + "="*60)
    print(f"TOP-{args.k} ENSEMBLE RESULTS (metric: {args.metric})")
    print("="*60)
    print(f"Ensemble AUC:        {ensemble_auc:.4f}")
    print(f"Ensemble F1:         {ensemble_f1:.4f}")
    print(f"Ensemble Recall:     {ensemble_recall:.4f}")
    print(f"Ensemble Precision:  {ensemble_precision:.4f}")
    print(f"Ensemble Composite:  {ensemble_composite:.4f}")
    print(f"Ensemble Prec-aware: {ensemble_precision_aware:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    print("\nBest Individual Model:")
    print(f"Seed {int(best_individual['seed'])} - {args.metric}: {best_individual[args.metric]:.4f}")
    print(f"AUC: {best_individual['auc']:.4f}, F1: {best_individual['f1']:.4f}")
    print(f"Precision-aware: {best_individual['precision_aware']:.4f}")

    # Calculate improvements
    auc_improvement = ensemble_auc - best_individual['auc']
    f1_improvement = ensemble_f1 - best_individual['f1']
    prec_aware_improvement = ensemble_precision_aware - best_individual['precision_aware']

    print("\nImprovements over best individual:")
    print(f"AUC:        {auc_improvement:+.4f}")
    print(f"F1:         {f1_improvement:+.4f}")
    print(f"Prec-aware: {prec_aware_improvement:+.4f}")

    # Save ensemble results
    results_file = os.path.join(args.weights_dir, f'ensemble_top{args.k}_results.json')
    import json
    results = {
        'ensemble_config': {
            'k': args.k,
            'metric': args.metric,
            'total_models': len(all_logits),
            'selected_seeds': top_k_seeds
        },
        'ensemble_metrics': {
            'auc': ensemble_auc,
            'f1': ensemble_f1,
            'recall': ensemble_recall,
            'precision': ensemble_precision,
            'composite': ensemble_composite,
            'precision_aware': ensemble_precision_aware,
            'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}
        },
        'best_individual': {
            'seed': int(best_individual['seed']),
            'auc': best_individual['auc'],
            'f1': best_individual['f1'],
            'precision_aware': best_individual['precision_aware']
        },
        'improvements': {
            'auc': auc_improvement,
            'f1': f1_improvement,
            'precision_aware': prec_aware_improvement
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nEnsemble results saved to: {results_file}")

if __name__ == '__main__':
    main()
