#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
import numpy as np
import tensorflow as tf

def load_helpers(this_dir: str):
    import importlib.machinery, importlib.util
    files = {
        'model': os.path.join(this_dir, 'model.py'),
        'utils': os.path.join(this_dir, 'utils.py'),
    }
    loaded = {}
    for name, path in files.items():
        loader = importlib.machinery.SourceFileLoader(name, path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)  # type: ignore
        loaded[name] = mod
    return loaded['model'], loaded['utils']

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='best.weights 路徑')
    ap.add_argument('--test_windows', required=True, help='測試 npz (windows_npz.npz)')
    ap.add_argument('--out', required=True, help='輸出 long_probs.npy 路徑')
    ap.add_argument('--mask_mode', choices=['soft','hard'], default='soft')
    ap.add_argument('--mask_threshold', type=float, default=0.6)
    ap.add_argument('--window_mask_min_mean', type=float, default=None)
    return ap.parse_args()

def main():
    args = parse_args()
    this_dir = os.path.dirname(__file__)
    model_mod, utils_mod = load_helpers(this_dir)
    build_cnn_bilstm = getattr(model_mod, 'build_cnn_bilstm')
    load_npz_windows = getattr(utils_mod, 'load_npz_windows')

    X_test, y_test = load_npz_windows(args.test_windows)
    y_test = y_test.reshape(-1).astype(int)
    if X_test.ndim != 3:
        raise RuntimeError(f'Unexpected X_test shape: {X_test.shape}')
    # Ensure (N,T,F) for Keras input (F,T)
    if X_test.shape[1] < X_test.shape[2]:
        input_shape = X_test.shape[1:]
        need_transpose = False
    else:
        input_shape = (X_test.shape[2], X_test.shape[1])
        X_test = np.transpose(X_test, (0,2,1))
        need_transpose = True

    # derive mask from utils-like helper in train file (simplified)
    def derive_mask(path: str):
        try:
            base = np.load(path, allow_pickle=True)
            feat_names = list(base['feature_list'].tolist()) if 'feature_list' in base else None
            F = len(feat_names) if feat_names is not None else None
            for key in ('long_raw','short_raw','long_norm','short_norm'):
                if key in base:
                    X_any = np.asarray(base[key]); break
            else:
                return None
            if X_any.ndim != 3: return None
            ch_axis = None
            if F is not None:
                if X_any.shape[1] == F: ch_axis = 1
                elif X_any.shape[2] == F: ch_axis = 2
            def take(x, idx, axis):
                return x[:, idx, :] if axis==1 else (x[:, :, idx] if axis==2 else None)
            if feat_names and 'occlusion_flag' in feat_names and ch_axis is not None:
                m = take(X_any, feat_names.index('occlusion_flag'), ch_axis)
                if m is not None: return np.clip(m,0,1).astype(np.float32)
            if ch_axis is not None:
                m = take(X_any, X_any.shape[ch_axis]-1, ch_axis)
                if m is not None: return np.clip(m,0,1).astype(np.float32)
        except Exception:
            return None
        return None

    m_test = derive_mask(args.test_windows)
    if m_test is not None and m_test.shape[1] != input_shape[1]:
        # align time
        T = input_shape[1]
        curT = m_test.shape[1]
        if curT > T:
            m_test = m_test[:, :T]
        else:
            pad = T - curT
            m_test = np.concatenate([m_test, np.zeros((m_test.shape[0], pad), dtype=m_test.dtype)], axis=1)

    model = build_cnn_bilstm(
        input_shape,
        num_filters=64, kernel_sizes=(3,5,3), conv_dropout=0.2,
        lstm_units=64, lstm_dropout=0.2, attn_units=32,
        use_mask=(m_test is not None), mask_mode=args.mask_mode, mask_threshold=float(args.mask_threshold),
    )
    model.load_weights(args.weights)
    if m_test is not None:
        probs = model.predict([X_test, m_test], verbose=0)
    else:
        probs = model.predict(X_test, verbose=0)
    if probs.ndim > 1 and probs.shape[-1] > 1:
        probs = probs[:,1]
    probs = probs.reshape(-1).astype('float32')

    # optional gating
    if (args.window_mask_min_mean is not None) and (m_test is not None):
        m_mean = m_test.mean(axis=1)
        gate = (m_mean >= float(args.window_mask_min_mean)).astype(probs.dtype)
        probs = probs * gate

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, probs)
    print('Saved long_probs:', args.out, 'shape=', probs.shape)

if __name__ == '__main__':
    main()
