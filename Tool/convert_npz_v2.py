#!/usr/bin/env python3
from __future__ import annotations
"""Convert legacy windows_dense_npz.npz (v1 / messy object arrays) into unified v2 schema.

目標 (v2) Schema 重點 (所有陣列 dtype=float32 / int64 / bool)：
- version: '2.0'
- feature_list: (F,)  (可選，若舊檔沒有則自動產生 f0..f{F-1})
- <split> in {short,long} 只在舊檔存在對應資料時才輸出；每個 split 可能包含：
    * {split}_raw:   (N, T, F)
    * {split}_norm:  (N, T, F)  (使用該 split 自身 mean/std 正規化，(x-mean)/std，std=0 -> 1)
    * {split}_mask:      (N, T, F) bool (目前全 True 或依 NaN 判斷)
    * {split}_mask_any:  (N, T) bool  (mask 任一維度為 True -> True)
    * {split}_label: (N,) int64
    * {split}_weight: (N,) float32 (若缺失 -> 全 1)
    * {split}_video_id, {split}_start_frame, {split}_end_frame  (可選 meta)
    * {split}_mean: (F,) float32  (raw 的全體特徵平均，忽略 NaN)
    * {split}_std:  (F,) float32  (raw 的全體特徵 std，忽略 NaN)
    * {split}_class_counts: (2,) int64  [neg, pos]

轉換邏輯：
1. 載入舊 npz。尋找鍵名 pattern：short_* / long_*。
2. 讀取 raw / norm：優先使用 *_raw 舊鍵；若只有 *_norm 仍可處理 (會反推 raw=norm * std + mean? -> 不做；直接將現有陣列當 raw，重新計算 norm)。
3. 處理 dtype=object：將每個元素轉 np.array 後 stack。
4. 修正方向：若得到 shape (N, F, T) 且 F(=36) < T 則轉置為 (N, T, F)。
5. 自動偵測 F (feature 維度)。若舊檔含 feature_list 則沿用；否則產生 f0..f{F-1}。
6. 產生 mask：若資料含 NaN -> mask=True 代表有效 (非 NaN)；否則全部 True。
7. 計算 mean/std (忽略 NaN)；std=0 位置改為 1 防除以 0。
8. 產生 norm=(raw-mean)/std。
9. label / weight 若缺就用預設；weight 缺 -> ones。
10. 產生 class_counts。
11. 儲存所有 split 陣列 + version + feature_list 到新的 npz。

使用：
python Tool/convert_npz_v2.py \
  --input /path/old/windows_dense_npz.npz \
  --output /path/new/windows_v2_all.npz \
  [--splits short long] \
  [--assume-feature-dim 36]

備註：若舊檔同時有 *_mask 而想保留，可加 --preserve-existing-mask。
"""

"""使用方式範例：
python Tool/convert_npz_v2.py 
--input /home/user/projects/train/train_data/slipce/windows_dense_npz.npz 
--output /home/user/projects/train/train_data/Slipce_2/windows_v2_all.npz 
--splits short long --assume-feature-dim 36 --preserve-existing-mask
"""
import argparse
import numpy as np
import os
from typing import List, Dict, Tuple


def _to_numpy_3d(arr, assume_feature_dim: int | None = None) -> np.ndarray:
    """Normalize incoming (legacy) array into shape (N, T, F) float32.

    Accept patterns:
    - object array: each element shape (T,F) or (F,T)
    - normal ndarray already in (N,T,F)
    - normal ndarray (N,F,T) -> transpose
    """
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        seqs = [np.asarray(x) for x in arr]
        arr = np.stack(seqs, axis=0)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array after stacking, got shape={arr.shape}")
    N, A, B = arr.shape
    # Heuristic for transpose
    # If assume_feature_dim provided, trust it.
    if assume_feature_dim is not None:
        if A == assume_feature_dim and B != assume_feature_dim:
            # shape (N, F, T) -> transpose
            arr = np.transpose(arr, (0, 2, 1))
        elif B == assume_feature_dim and A != assume_feature_dim:
            # already (N, T, F)
            pass
        elif A == assume_feature_dim and B == assume_feature_dim:
            # Ambiguous (T == F); assume already (N,T,F)
            pass
        else:
            # If neither dim matches assume_feature_dim, try guess: smaller dim is feature
            pass
    else:
        # Guess: feature dimension likely smaller (like 36) and time larger (like 75)
        if A <= B:  # assume A=F <= T=B -> transpose needed if currently (N,F,T)
            # If samples appear constant across time when not transposed, easier to just check typical numbers later; keep rule simple.
            # We'll check by example variance heuristic:
            if A < B:  # typical case F(36) < T(75)
                # If currently (N, F, T) we transpose to (N, T, F)
                # Check if axis1 variance > axis2 variance to decide? Keep deterministic: transpose.
                arr = np.transpose(arr, (0, 2, 1))  # (N,T,F)
        else:
            # A > B, then B likely feature
            arr = np.transpose(arr, (0, 2, 1))
    arr = arr.astype(np.float32)
    return arr


def _compute_stats(x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # x: (N,T,F), mask: (N,T,F) bool where True means valid
    valid = mask
    # Replace invalid with NaN for np.nanmean
    x_masked = np.where(valid, x, np.nan)
    mean = np.nanmean(x_masked, axis=(0, 1))  # (F,)
    std = np.nanstd(x_masked, axis=(0, 1))
    # Handle all-NaN (becomes NaN): replace with 0 / 1 std
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=0.0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _make_mask(x: np.ndarray) -> np.ndarray:
    # Valid if not NaN
    if np.isnan(x).any():
        return ~np.isnan(x)
    # All valid
    return np.ones_like(x, dtype=bool)


def process_split(data: Dict[str, np.ndarray], split: str, assume_feature_dim: int | None, preserve_existing_mask: bool):
    out: Dict[str, np.ndarray] = {}
    raw_key_candidates = [f'{split}_raw', f'{split}_norm']  # fallback to norm as raw if raw missing
    src_raw_key = None
    for k in raw_key_candidates:
        if k in data:
            src_raw_key = k
            break
    if src_raw_key is None:
        # Nothing for this split
        return out
    raw = _to_numpy_3d(data[src_raw_key], assume_feature_dim)
    if preserve_existing_mask and f'{split}_mask' in data:
        mask = data[f'{split}_mask']
        # Align shape
        mask = _to_numpy_3d(mask.astype(np.float32), assume_feature_dim) > 0.5
    else:
        mask = _make_mask(raw)
    if mask.shape != raw.shape:
        raise ValueError(f"Mask shape mismatch {mask.shape} vs raw {raw.shape}")
    mask_any = np.any(mask, axis=2)  # (N,T)
    mean, std = _compute_stats(raw, mask)
    norm = (raw - mean) / std
    # 將無效 (mask=False) 位置填 0，避免後續訓練 NaN 傳播
    invalid = ~mask
    if invalid.any():
        norm[invalid] = 0.0
    # labels
    label_key = f'{split}_label'
    if label_key in data:
        labels = data[label_key].astype(np.int64)
    else:
        labels = np.zeros((raw.shape[0],), dtype=np.int64)
    weight_key = f'{split}_weight'
    if weight_key in data:
        weights = data[weight_key].astype(np.float32)
    else:
        weights = np.ones((raw.shape[0],), dtype=np.float32)
    neg = int((labels == 0).sum())
    pos = int((labels == 1).sum())
    class_counts = np.array([neg, pos], dtype=np.int64)
    out[f'{split}_raw'] = raw
    out[f'{split}_norm'] = norm.astype(np.float32)
    out[f'{split}_mask'] = mask.astype(np.bool_)
    out[f'{split}_mask_any'] = mask_any.astype(np.bool_)
    out[f'{split}_label'] = labels
    out[f'{split}_weight'] = weights
    out[f'{split}_mean'] = mean
    out[f'{split}_std'] = std
    out[f'{split}_class_counts'] = class_counts
    # meta fields
    for mk in [f'{split}_video_id', f'{split}_start_frame', f'{split}_end_frame']:
        if mk in data:
            out[mk] = data[mk]
    return out


def main():
    ap = argparse.ArgumentParser(description='Convert legacy windows_dense_npz.npz to v2 schema')
    ap.add_argument('--input', required=True, help='Path to legacy npz')
    ap.add_argument('--output', required=True, help='Output npz path (will overwrite)')
    ap.add_argument('--splits', nargs='*', default=['short', 'long'], help='Which splits to attempt')
    ap.add_argument('--assume-feature-dim', type=int, default=36, help='Feature dimension hint (set 0 to disable)')
    ap.add_argument('--preserve-existing-mask', action='store_true', help='Use existing *_mask if present')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f'Input not found: {args.input}')
    data = np.load(args.input, allow_pickle=True)
    out: Dict[str, np.ndarray] = {}

    # Copy feature_list or synthesize later
    feature_list = None
    if 'feature_list' in data:
        feature_list = data['feature_list']
    # Process each split
    assume_dim = args.assume_feature_dim if args.assume_feature_dim > 0 else None
    any_split = False
    for sp in args.splits:
        part = process_split(data, sp, assume_dim, args.preserve_existing_mask)
        if part:
            any_split = True
            out.update(part)
    if not any_split:
        raise SystemExit('No splits converted (check --splits or input file keys).')

    # Infer feature dim from one of the converted arrays
    sample_key = None
    for k in ['short_raw', 'long_raw']:
        if k in out:
            sample_key = k
            break
    assert sample_key is not None
    F = out[sample_key].shape[2]
    if feature_list is None:
        feature_list = np.array([f'f{i}' for i in range(F)], dtype=object)

    out['feature_list'] = feature_list
    out['version'] = np.array(['2.0'], dtype=object)

    # Save
    np.savez_compressed(args.output, **out)
    print(f'[OK] Wrote v2 dataset to {args.output}')
    for sp in args.splits:
        if f'{sp}_raw' in out:
            cc = out[f'{sp}_class_counts']
            print(f'  Split {sp}: shape={out[f"{sp}_raw"].shape} class_counts={cc.tolist()}')


if __name__ == '__main__':
    main()
