#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble inference for K-fold CNN+BiLSTM attention model.

只進行推論：讀取 kfold 結果資料夾內各 fold_*/best.weights，
對指定 test set 做概率平均，輸出 ensemble_test_results.json。
可選擇輸出每個 fold 及 ensemble 的原始概率 .npy。

Usage (example):

python ensemble_infer.py \
  --kfold_root /home/user/projects/train/model/long_window/CNN_BiLSTM/result_kfold \
  --test_windows /home/user/projects/train/test_data/slipce_thresh040/windows_npz.npz

附加選項：
  --windows / --val_windows  (僅為 mask 對齊時可能需要 train/val 來源; 若不提供，只用 test)
  --mask_mode soft|hard
  --mask_threshold 0.6
  --window_mask_min_mean 0.5  (若設定，會套用 gating)
  --overwrite  (允許覆寫已存在 ensemble_test_results.json)
  --save_probs  (另存 fold 概率與 ensemble 概率 .npy)
"""
from __future__ import annotations
import os, argparse, json, sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

# Import local modules (model builder & helpers)
try:
    from model import build_cnn_bilstm  # type: ignore
    from utils import load_npz_windows  # type: ignore
except Exception:
    # Fallback: direct load model.py & utils.py via SourceFileLoader
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
    p.add_argument('--kfold_root', required=True, help='根目錄，底下包含 fold_01, fold_02 ... 以及 final_all_train')
    p.add_argument('--test_windows', required=True, help='Test NPZ (windows_npz.npz)')
    p.add_argument('--windows', default=None, help='(可選) train windows，用於 mask 對齊 (若需要)')
    p.add_argument('--val_windows', default=None, help='(可選) val windows，用於 mask 對齊 (若需要)')
    p.add_argument('--mask_mode', choices=['soft','hard'], default='soft')
    p.add_argument('--mask_threshold', type=float, default=0.6)
    p.add_argument('--window_mask_min_mean', type=float, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--save_probs', action='store_true')
    return p.parse_args(argv)


def derive_mask(path: str):
    """複製 train_cnn_bilstm 中的 mask 擷取邏輯 (簡化版)。
    回傳 (N,T) 或 None。
    """
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
                ch_axis = 1  # (N,F,T)
            elif X_any.shape[2] == F:
                ch_axis = 2  # (N,T,F)
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


def align_time(arr, target_T):
    if arr.shape[2] == target_T:
        return arr
    cur_T = arr.shape[2]
    if cur_T > target_T:
        return arr[:, :, :target_T]
    pad = target_T - cur_T
    pad_block = np.zeros((arr.shape[0], arr.shape[1], pad), dtype=arr.dtype)
    return np.concatenate([arr, pad_block], axis=2)


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


def evaluate_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = float(roc_auc_score(y_true, probs))
    f1 = float(f1_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    composite = 0.5 * recall + 0.3 * f1 + 0.2 * auc
    precision_aware = 0.5 * precision + 0.3 * f1 + 0.2 * auc
    return dict(auc=auc, f1=f1, recall=recall, precision=float(precision), composite=composite, precision_aware=precision_aware, TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


def main():
    args = parse_args()

    final_dir = os.path.join(args.kfold_root, 'final_all_train')
    out_json = os.path.join(final_dir, 'ensemble_test_results.json')
    if os.path.exists(out_json) and (not args.overwrite):
        print('[ensemble] 已存在，使用 --overwrite 以重新計算:', out_json)
        return

    # 載入 test windows
    X_test, y_test = load_npz_windows(args.test_windows)
    y_test = y_test.flatten().astype(int)
    # 判斷 (N,F,T) or (N,T,F)
    if X_test.ndim != 3:
        print('[ensemble] ERROR test rank', X_test.shape)
        return
    if X_test.shape[1] < X_test.shape[2]:
        input_shape = X_test.shape[1:]  # (F,T)
        need_transpose = False
    else:
        input_shape = (X_test.shape[2], X_test.shape[1])
        X_test = np.transpose(X_test, (0,2,1))
        need_transpose = True

    # 取得 mask (test)
    m_test = derive_mask(args.test_windows)
    if m_test is not None and m_test.shape[1] != input_shape[1]:
        m_test = align_mask(m_test, input_shape[1])

    # 收集 fold weights
    fold_dirs = sorted([d for d in os.listdir(args.kfold_root) if d.startswith('fold_')])
    if not fold_dirs:
        print('[ensemble] 沒找到 fold_* 目錄')
        return
    probs_list = []
    fold_used = 0
    for fd in fold_dirs:
        weight_path = os.path.join(args.kfold_root, fd, 'best.weights')
        if not os.path.exists(weight_path):
            print('[ensemble] skip', fd, '(no best.weights)')
            continue
        # 建立模型 (推論用; use_mask 取決於 m_test 是否存在)
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
            print('[ensemble] load fail', fd, e)
            continue
        if m_test is not None:
            p = model.predict([X_test, m_test], verbose=0)
        else:
            p = model.predict(X_test, verbose=0)
        if p.ndim > 1 and p.shape[-1] > 1:
            p = p[:,1]
        p = p.reshape(-1)
        probs_list.append(p)
        fold_used += 1
        print(f'[ensemble] fold {fd} done, shape={p.shape}')

    if not probs_list:
        print('[ensemble] 無可用 fold 概率')
        return

    ens_probs = np.mean(np.stack(probs_list, axis=0), axis=0)

    # gating
    if args.window_mask_min_mean is not None and m_test is not None:
        try:
            m_mean = m_test.mean(axis=1)
            gate = (m_mean >= float(args.window_mask_min_mean)).astype(ens_probs.dtype)
            ens_probs = ens_probs * gate
        except Exception:
            pass

    metrics = evaluate_metrics(y_test, ens_probs, 0.5)
    result = {
        'ensemble_test': metrics,
        'folds_used': fold_used,
        'params': {
            'mask_mode': args.mask_mode,
            'mask_threshold': float(args.mask_threshold),
            'window_mask_min_mean': (float(args.window_mask_min_mean) if args.window_mask_min_mean is not None else None)
        }
    }
    os.makedirs(final_dir, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print('[ensemble] saved ->', out_json)

    if args.save_probs:
        np.save(os.path.join(final_dir, 'ensemble_probs.npy'), ens_probs)
        for i, p in enumerate(probs_list, start=1):
            np.save(os.path.join(final_dir, f'fold{i:02d}_probs.npy'), p)
        print('[ensemble] saved raw probability arrays')


if __name__ == '__main__':
    main()
