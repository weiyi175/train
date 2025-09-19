#!/usr/bin/env python3
"""
Deployable inference script for TOP-10 logit-averaging ensemble.
Creates probability outputs, default predictions and suggests thresholds.
Usage: python deploy_ensemble_top10.py --weights_dir result_ensemble_weights
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
import tensorflow as tf
from model import build_cnn_bilstm
from utils import load_npz_windows
from scipy.optimize import minimize


def load_model_with_weights(model, weights_path):
    try:
        model.load_weights(weights_path)
        return True
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")
        return False


def suggest_thresholds(y_true, probs):
    # Evaluate a grid of thresholds and choose useful suggestions
    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        auc = roc_auc_score(y_true, probs)
        prec_aware = 0.5 * prec + 0.3 * f1 + 0.2 * auc
        rows.append((float(t), float(prec), float(rec), float(f1), float(prec_aware)))

    df = pd.DataFrame(rows, columns=['threshold', 'precision', 'recall', 'f1', 'precision_aware'])
    # threshold that maximizes precision_aware
    t_pa = df.sort_values('precision_aware', ascending=False).iloc[0]
    # threshold that maximizes F1
    t_f1 = df.sort_values('f1', ascending=False).iloc[0]
    # threshold that yields highest precision while recall >= 0.7 (if any)
    df_rec = df[df.recall >= 0.7]
    if len(df_rec) > 0:
        t_prec_rec = df_rec.sort_values('precision', ascending=False).iloc[0]
    else:
        t_prec_rec = df.sort_values('precision', ascending=False).iloc[0]

    suggestions = {
        'threshold_max_precision_aware': t_pa.to_dict(),
        'threshold_max_f1': t_f1.to_dict(),
        'threshold_max_precision_with_rec>=0.7': t_prec_rec.to_dict()
    }
    return suggestions, df


def temperature_scale_probs(logits, temp):
    # logits assumed to be raw model outputs in [0,1] as probabilities; convert to logit space
    eps = 1e-7
    clipped = np.clip(logits, eps, 1 - eps)
    logit = np.log(clipped / (1.0 - clipped))
    scaled = 1.0 / (1.0 + np.exp(-logit / float(temp)))
    return scaled


def fit_temperature(y_true, probs, initial_temp=1.0):
    # Optimize temperature by minimizing negative log-likelihood (binary crossentropy)
    eps = 1e-12

    def nll(temp):
        if temp[0] <= 0:
            return 1e6
        p = temperature_scale_probs(probs, temp[0])
        p = np.clip(p, eps, 1 - eps)
        loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        return float(loss)

    res = minimize(nll, x0=np.array([initial_temp]), bounds=[(1e-3, 10.0)])
    if res.success:
        return float(res.x[0])
    else:
        return initial_temp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_dir', type=str, default='result_ensemble_weights')
    parser.add_argument('--results_json', type=str, help='Optional results json (top-k) to read selected seeds')
    parser.add_argument('--windows', type=str,
                        default='/home/user/projects/train/train_data/slipce/windows_npz.npz')
    parser.add_argument('--calib_windows', type=str, default=None,
                        help='Optional separate NPZ file to use for temperature calibration')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to save outputs (defaults to weights_dir)')
    args = parser.parse_args()

    weights_dir = args.weights_dir
    out_dir = args.output_dir or weights_dir
    os.makedirs(out_dir, exist_ok=True)

    # Load selected seeds from results json if provided, otherwise try ensemble_top10_results.json
    selected_seeds = None
    if args.results_json:
        rpath = args.results_json
    else:
        rpath = os.path.join(weights_dir, 'ensemble_top10_results.json')

    if os.path.exists(rpath):
        with open(rpath, 'r') as f:
            data = json.load(f)
        selected_seeds = data.get('ensemble_config', {}).get('selected_seeds', None)

    if selected_seeds is None:
        # fallback: take top-10 seed folders by name (seed1..seed50 sorted) and pick first 10
        seed_dirs = sorted([d for d in os.listdir(weights_dir) if d.startswith('seed')])
        selected_seeds = [int(d.replace('seed', '')) for d in seed_dirs[:10]]

    print(f"Selected seeds for deployment: {selected_seeds}")

    # Load test data (used here as example input for saving probs)
    print('Loading data...')
    X_test, y_test = load_npz_windows(args.windows)
    m_test = None

    # Optionally load a separate calibration (validation) set for temperature fitting
    use_calib = False
    X_calib = None
    y_calib = None
    if args.calib_windows:
        if os.path.exists(args.calib_windows):
            print(f'Loading calibration data from {args.calib_windows}...')
            X_calib, y_calib = load_npz_windows(args.calib_windows)
            use_calib = True
        else:
            print(f'Calibration file {args.calib_windows} not found; falling back to main windows for calibration')

    # Build model template (same logic as ensemble script)
    if X_test.shape[1] == 36 and X_test.shape[2] == 75:
        input_shape = (36, 75)
    elif X_test.shape[1] == 36 and X_test.shape[2] == 30:
        input_shape = (36, 30)
    else:
        input_shape = (X_test.shape[1], X_test.shape[2])

    model_template = build_cnn_bilstm(
        input_shape=input_shape,
        num_filters=64,
        kernel_sizes=(3, 5, 3),
        conv_dropout=0.2,
        lstm_units=64,
        lstm_dropout=0.2,
        attn_units=32,
        use_mask=False
    )

    all_logits_test = []
    all_logits_calib = [] if use_calib else None
    for seed in selected_seeds:
        seed_dir = f'seed{seed}'
        weights_path = os.path.join(weights_dir, seed_dir, '01', 'final.weights')
        weights_index = weights_path + '.index'
        if not os.path.exists(weights_index):
            print(f'Warning: missing weights for seed {seed}: {weights_index} - skipping')
            continue

        model = tf.keras.models.clone_model(model_template)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        ok = load_model_with_weights(model, weights_path)
        if not ok:
            continue

        # predict on test
        if m_test is not None:
            logits_test = model.predict([X_test, m_test], verbose=0, batch_size=32)
        else:
            logits_test = model.predict(X_test, verbose=0, batch_size=32)

        # predict on calib if available
        if use_calib:
            if m_test is not None:
                logits_calib = model.predict([X_calib, m_test], verbose=0, batch_size=32)
            else:
                logits_calib = model.predict(X_calib, verbose=0, batch_size=32)

        # normalize shapes
        if logits_test.ndim > 1 and logits_test.shape[-1] > 1:
            logits_test = logits_test[:, 1]
        elif logits_test.ndim > 1:
            logits_test = logits_test.squeeze()

        all_logits_test.append(logits_test)

        if use_calib:
            if logits_calib.ndim > 1 and logits_calib.shape[-1] > 1:
                logits_calib = logits_calib[:, 1]
            elif logits_calib.ndim > 1:
                logits_calib = logits_calib.squeeze()
            all_logits_calib.append(logits_calib)
        print(f'Loaded seed {seed}')

    if len(all_logits_test) == 0:
        print('No models loaded; aborting')
        return

    ensemble_logits_test = np.mean(all_logits_test, axis=0)
    if use_calib and len(all_logits_calib) > 0:
        ensemble_logits_calib = np.mean(all_logits_calib, axis=0)
    else:
        ensemble_logits_calib = None
    # Save probabilities and default predictions (threshold 0.5)
    probs_path = os.path.join(out_dir, 'ensemble_top10_probs.npy')
    preds_path = os.path.join(out_dir, 'ensemble_top10_preds_0.5.npy')
    np.save(probs_path, ensemble_logits_test)
    np.save(preds_path, (ensemble_logits_test >= 0.5).astype(int))

    # Suggest thresholds
    suggestions, df_grid = suggest_thresholds(y_test, ensemble_logits_test)

    # Compute metrics at default 0.5 for quick report
    ensemble_auc = float(roc_auc_score(y_test, ensemble_logits_test))
    ensemble_preds = (ensemble_logits_test >= 0.5).astype(int)
    ensemble_f1 = float(f1_score(y_test, ensemble_preds))
    ensemble_rec = float(recall_score(y_test, ensemble_preds))
    ensemble_prec = float(precision_score(y_test, ensemble_preds))

    meta = {
        'selected_seeds': selected_seeds,
        'saved_probs': probs_path,
        'saved_preds_0.5': preds_path,
        'metrics_at_0.5': {
            'auc': ensemble_auc,
            'f1': ensemble_f1,
            'recall': ensemble_rec,
            'precision': ensemble_prec
        },
    'threshold_suggestions': suggestions
    }

    meta_path = os.path.join(out_dir, 'deploy_ensemble_top10_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Save grid as CSV for analysis
    grid_path = os.path.join(out_dir, 'deploy_threshold_grid.csv')
    df_grid.to_csv(grid_path, index=False)

    # Fit temperature on the provided (validation/test) set and save calibrated probabilities
    try:
        # decide which data to use for temperature fitting: calibration set (if provided) or test
        if ensemble_logits_calib is not None:
            print('Fitting temperature on calibration set')
            fitted_temp = fit_temperature(y_calib, ensemble_logits_calib, initial_temp=1.0)
            meta['temperature_calibration_source'] = args.calib_windows
        else:
            print('Fitting temperature on main windows (default)')
            fitted_temp = fit_temperature(y_test, ensemble_logits_test, initial_temp=1.0)
            meta['temperature_calibration_source'] = args.windows

        # apply fitted temperature to test probabilities
        calibrated_probs = temperature_scale_probs(ensemble_logits_test, fitted_temp)
        calibrated_probs_path = os.path.join(out_dir, 'ensemble_top10_probs_calibrated.npy')
        calibrated_preds_path = os.path.join(out_dir, f'ensemble_top10_preds_calibrated_{fitted_temp:.3f}.npy')
        np.save(calibrated_probs_path, calibrated_probs)
        np.save(calibrated_preds_path, (calibrated_probs >= 0.5).astype(int))

        # calibrated metrics
        cal_auc = float(roc_auc_score(y_test, calibrated_probs))
        cal_preds = (calibrated_probs >= 0.5).astype(int)
        cal_f1 = float(f1_score(y_test, cal_preds))
        cal_rec = float(recall_score(y_test, cal_preds))
        cal_prec = float(precision_score(y_test, cal_preds))

        meta.update({
            'temperature_calibration': {
                'fitted_temperature': float(fitted_temp),
                'calibrated_probs': calibrated_probs_path,
                'calibrated_preds_0.5': calibrated_preds_path,
                'calibrated_metrics_at_0.5': {
                    'auc': cal_auc,
                    'f1': cal_f1,
                    'recall': cal_rec,
                    'precision': cal_prec
                }
            }
        })

        # rewrite meta with calibration info
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'Calibration applied: temperature={fitted_temp:.4f}; calibrated probs saved to {calibrated_probs_path}')
    except Exception as e:
        print(f'Calibration failed: {e}')

    print('\nDeployment artifacts:')
    print(f' - probabilities: {probs_path}')
    print(f' - default preds (0.5): {preds_path}')
    print(f' - meta: {meta_path}')
    print(f' - threshold grid CSV: {grid_path}')


if __name__ == '__main__':
    main()
