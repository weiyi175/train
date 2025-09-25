#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold scanner for final_all_train model (or any single model weights) to optimize F1 or precision-aware metric.

流程:
 1. 載入 test set (或指定 windows 作為評估)
 2. 載入 final_all_train 目錄下的權重: 優先 best.weights (若存在) 否則 final.weights
 3. 產生概率 -> 掃描 [0,1] 區間 (step) 計算指標
 4. 找出: F1 最佳閾值, precision_aware 最佳閾值, 以及同時輸出完整曲線 (可選保存 .csv/.npy)

Usage:
python threshold_scan.py \
  --final_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_kfold/final_all_train \
  --test_windows /home/user/projects/train/test_data/slipce_thresh040/windows_npz.npz \
  --step 0.01 --save_curve
"""
from __future__ import annotations
import os, sys, argparse, json, csv
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

try:
    from model import build_cnn_bilstm  # type: ignore
    from utils import load_npz_windows  # type: ignore
except Exception:
    import importlib.machinery, importlib.util
    this_dir = os.path.dirname(__file__)
    files = {
        'model': os.path.join(this_dir, 'model.py'),
        'utils': os.path.join(this_dir, 'utils.py'),
    }
    loaded = {}
    for name, path in files.items():
        if os.path.exists(path):
            loader = importlib.machinery.SourceFileLoader(name, path)
            spec = importlib.util.spec_from_loader(loader.name, loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)  # type: ignore
            loaded[name] = mod
    if 'model' in loaded and hasattr(loaded['model'], 'build_cnn_bilstm'):
        build_cnn_bilstm = getattr(loaded['model'], 'build_cnn_bilstm')  # type: ignore
    else:
        raise ImportError('Cannot load build_cnn_bilstm from model.py')
    if 'utils' in loaded and hasattr(loaded['utils'], 'load_npz_windows'):
        load_npz_windows = getattr(loaded['utils'], 'load_npz_windows')  # type: ignore
    else:
        raise ImportError('Cannot load load_npz_windows from utils.py')


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--final_dir', required=True, help='final_all_train 目錄 (含 best.weights 或 final.weights)')
    p.add_argument('--test_windows', required=True, help='Test NPZ (windows_npz.npz)')
    p.add_argument('--mask_mode', choices=['soft','hard'], default='soft')
    p.add_argument('--mask_threshold', type=float, default=0.6)
    p.add_argument('--window_mask_min_mean', type=float, default=None)
    p.add_argument('--step', type=float, default=0.01, help='掃描步長 (e.g. 0.005)')
    p.add_argument('--save_curve', action='store_true', help='輸出完整掃描結果 threshold_curve.csv / .npy')
    p.add_argument('--save_probs', action='store_true', help='保存原始機率 test_probs.npy')
    return p.parse_args(argv)


def derive_mask(path: str):
    try:
        if path is None or (not os.path.exists(path)):
            return None
        base = np.load(path, allow_pickle=True)
        feat_names = list(base['feature_list'].tolist()) if 'feature_list' in base else None
        F = len(feat_names) if feat_names is not None else None
        for key in ('long_raw', 'short_raw', 'long_norm', 'short_norm'):
            if key in base:
                X_any = np.asarray(base[key])
                break
        else:
            return None
        if X_any.ndim != 3:
            return None
        ch_axis = None
        if F is not None:
            if X_any.shape[1] == F:
                ch_axis = 1
            elif X_any.shape[2] == F:
                ch_axis = 2
        def take_channel(x, idx, axis):
            if axis == 1:
                return x[:, idx, :]
            elif axis == 2:
                return x[:, :, idx]
            return None
        if feat_names and 'occlusion_flag' in feat_names and ch_axis is not None:
            ch = feat_names.index('occlusion_flag')
            m = take_channel(X_any, ch, ch_axis)
            if m is not None:
                return np.clip(m,0,1).astype(np.float32)
        if ch_axis is not None:
            m = take_channel(X_any, X_any.shape[ch_axis]-1, ch_axis)
            if m is not None:
                return np.clip(m,0,1).astype(np.float32)
    except Exception as e:
        print('[mask] derive fail', path, e)
    return None


def align_mask(mask, target_T):
    if mask is None:
        return None
    if mask.shape[1] == target_T:
        return mask
    cur_T = mask.shape[1]
    if cur_T > target_T:
        return mask[:, :target_T]
    pad = target_T - cur_T
    return np.concatenate([mask, np.zeros((mask.shape[0], pad), dtype=mask.dtype)], axis=1)


def compute_metrics(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    auc = float(roc_auc_score(y_true, probs))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    composite = 0.5 * recall + 0.3 * f1 + 0.2 * auc
    precision_aware = 0.5 * precision + 0.3 * f1 + 0.2 * auc
    return dict(threshold=float(thr), auc=auc, f1=f1, recall=recall, precision=float(precision), composite=composite, precision_aware=precision_aware, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def main():
    args = parse_args()
    weight_best = os.path.join(args.final_dir, 'best.weights')
    weight_final = os.path.join(args.final_dir, 'final.weights')
    if os.path.exists(weight_best):
        weight_path = weight_best
    elif os.path.exists(weight_final):
        weight_path = weight_final
    else:
        print('[threshold_scan] 找不到 best.weights 或 final.weights 於', args.final_dir)
        return
    print('[threshold_scan] 使用權重:', weight_path)

    X_test, y_test = load_npz_windows(args.test_windows)
    y_test = y_test.flatten().astype(int)
    if X_test.ndim != 3:
        print('[threshold_scan] ERROR test rank', X_test.shape)
        return
    if X_test.shape[1] < X_test.shape[2]:
        input_shape = X_test.shape[1:]
    else:
        input_shape = (X_test.shape[2], X_test.shape[1])
        X_test = np.transpose(X_test, (0,2,1))

    m_test = derive_mask(args.test_windows)
    if m_test is not None and m_test.shape[1] != input_shape[1]:
        m_test = align_mask(m_test, input_shape[1])

    model = build_cnn_bilstm(
        input_shape,
        num_filters=64,
        kernel_sizes=(3,5,3),
        conv_dropout=0.2,
        lstm_units=64,
        lstm_dropout=0.2,
        attn_units=32,
        use_mask=(m_test is not None),
        mask_mode=args.mask_mode,
        mask_threshold=float(args.mask_threshold),
    )
    try:
        model.load_weights(weight_path)
    except Exception as e:
        print('[threshold_scan] load fail', e)
        return

    if m_test is not None:
        probs = model.predict([X_test, m_test], verbose=0)
    else:
        probs = model.predict(X_test, verbose=0)
    if probs.ndim > 1 and probs.shape[-1] > 1:
        probs = probs[:,1]
    probs = probs.reshape(-1)

    if args.window_mask_min_mean is not None and m_test is not None:
        try:
            m_mean = m_test.mean(axis=1)
            gate = (m_mean >= float(args.window_mask_min_mean)).astype(probs.dtype)
            probs = probs * gate
        except Exception:
            pass

    if args.save_probs:
        np.save(os.path.join(args.final_dir, 'test_probs.npy'), probs)

    step = float(args.step)
    thrs = np.arange(0.0, 1.0 + 1e-9, step)
    curve = []
    best_f1 = (-1, None)
    best_precaware = (-1, None)
    # AUC fixed for all thresholds, compute once
    global_auc = float(roc_auc_score(y_test, probs))

    for t in thrs:
        metrics = compute_metrics(y_test, probs, t)
        # override auc to be consistent (floating diff is minor but keep single calc if desired)
        metrics['auc'] = global_auc
        curve.append(metrics)
        if metrics['f1'] > best_f1[0]:
            best_f1 = (metrics['f1'], metrics)
        if metrics['precision_aware'] > best_precaware[0]:
            best_precaware = (metrics['precision_aware'], metrics)

    out = {
        'weight_used': os.path.basename(weight_path),
        'global_auc': global_auc,
        'best_f1': best_f1[1],
        'best_precision_aware': best_precaware[1],
        'scan_step': step,
        'num_points': len(curve)
    }
    with open(os.path.join(args.final_dir, 'threshold_scan_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('[threshold_scan] saved -> threshold_scan_results.json')

    if args.save_curve:
        curve_path_csv = os.path.join(args.final_dir, 'threshold_curve.csv')
        curve_path_npy = os.path.join(args.final_dir, 'threshold_curve.npy')
        # CSV
        keys = list(curve[0].keys()) if curve else []
        with open(curve_path_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in curve:
                w.writerow(row)
        np.save(curve_path_npy, curve)
        print('[threshold_scan] curve saved ->', curve_path_csv, curve_path_npy)


if __name__ == '__main__':
    main()
