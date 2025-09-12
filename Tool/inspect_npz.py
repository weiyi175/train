#!/usr/bin/env python3
"""Utility to inspect windows_dense_npz.npz and test WindowsNPZDataset.

Example:
  python Tool/inspect_npz.py \
      --npz train_data/slipce/windows_dense_npz.npz \
      --splits short long --head 2
"""
"""
用途：快速檢查 windows_dense_npz.npz 內容與可讀性。
主要功能：

開啟指定 NPZ，列出全部 key、shape、dtype。
對指定 splits（預設 short、long）建立 WindowsNPZDataset。
顯示每個 split 的樣本數、正負類分佈、權重統計 (min/max/mean)。
抽前 N 筆（--head）列印：x / mask 形狀、label、weight、是否含 NaN、有效幀比例。
匯入路徑有兩段嘗試（Tool.dataset_npz → dataset_npz）避免路徑問題。
用法範例：
python inspect_npz.py --npz windows_dense_npz.npz --splits short long --head 2
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from typing import Sequence


def list_keys(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as d:
        keys = sorted(d.keys())
        print(f"[NPZ] key count: {len(keys)}")
        for k in keys:
            arr = d[k]
            if hasattr(arr, 'shape'):
                print(f"  {k:25s} shape={arr.shape} dtype={getattr(arr,'dtype',type(arr))}")
            else:
                print(f"  {k:25s} type={type(arr)}")


def dataset_summary(npz_path: str, splits: Sequence[str], head: int):
    import importlib, traceback
    try:
        ds_mod = importlib.import_module('Tool.dataset_npz', package=None)
    except ModuleNotFoundError:
        # fallback relative
        try:
            ds_mod = importlib.import_module('dataset_npz')
        except Exception:  # pragma: no cover
            print('[ERROR] Could not import dataset_npz')
            traceback.print_exc()
            return
    WindowsNPZDataset = getattr(ds_mod, 'WindowsNPZDataset')
    for split in splits:
        try:
            ds = WindowsNPZDataset(npz_path, split=split, use_norm=True, temporal_jitter_frames=0)
        except Exception as e:
            print(f"[Dataset] {split} INIT ERROR: {e}")
            continue
        n = len(ds)
        print(f"[Dataset] {split}: length={n}")
        if n == 0:
            continue
        # label distribution & weight stats
        ys = [int(ds.y[i]) for i in ds.keep_idx]
        ys_arr = np.array(ys)
        pos = int((ys_arr==1).sum()); neg = int((ys_arr==0).sum())
        w_arr = np.array([float(ds.weight[i]) for i in ds.keep_idx], dtype=np.float32)
        print(f"  labels: neg={neg} pos={pos}")
        print(f"  weight stats: min={w_arr.min():.4f} max={w_arr.max():.4f} mean={w_arr.mean():.4f}")
        # sample(s)
        for j in range(min(head, n)):
            s = ds[j]
            x, m, y = s['x'], s['mask'], s['y']
            w = s['weight']
            print(f"  sample[{j}] x={tuple(x.shape)} mask={tuple(m.shape)} y={y.item()} w={float(w):.4f} NaN?={bool(np.isnan(x.numpy()).any())}")
            if m.shape[0] != x.shape[0]:
                print("    [WARN] time length mismatch between x and mask")
            valid_ratio = float(m.numpy().any(axis=1).mean()) if m.ndim==2 else float(m.numpy().mean())
            print(f"    valid frame ratio≈{valid_ratio:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='Path to windows_dense_npz.npz')
    ap.add_argument('--splits', nargs='+', default=['short','long'], help='Which splits to inspect')
    ap.add_argument('--head', type=int, default=1, help='How many samples per split to show')
    args = ap.parse_args()
    npz_path = os.path.abspath(args.npz)
    if not os.path.exists(npz_path):
        raise SystemExit(f"NPZ not found: {npz_path}")
    print('[Inspect] NPZ:', npz_path)
    list_keys(npz_path)
    dataset_summary(npz_path, args.splits, args.head)


if __name__ == '__main__':  # pragma: no cover
    main()
