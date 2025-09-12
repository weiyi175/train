#!/usr/bin/env python3
"""PyTorch Dataset/DataLoader utilities for windows_dense_npz.npz

Features:
- Loads dense NPZ with meta indices and masks.
- Exposes (x, y, weight, meta) with choice of raw or normalized inputs.
- Optional on-the-fly temporal jitter by random time roll within ±k frames.
- Helper to build DataLoader with WeightedRandomSampler using per-sample weights
  (combines class-balance and provided 'short_weight').

Example:
    from dataset_npz import WindowsNPZDataset, build_dataloader
    ds = WindowsNPZDataset('/path/windows_dense_npz.npz', split='short', use_norm=True, temporal_jitter_frames=2)
    dl = build_dataloader(ds, batch_size=64, num_workers=2, balance_by_class=True)
    for batch in dl:
        x = batch['x']         # (B, T, F)
        y = batch['y']         # (B,)
        w = batch['weight']    # (B,)
        meta = batch['meta']   # dict of arrays
"""
from __future__ import annotations
import os
import math
import numpy as np
from typing import Dict, Any, Optional

# Lazy import torch to avoid hard dependency at import time
torch = None  # type: ignore
def _ensure_torch():
    global torch
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')
    return torch


class WindowsNPZDataset(object):
    def __init__(self,
                 npz_path: str,
                 split: str = 'short',  # 'short' or 'long'
                 use_norm: bool = True,
                 temporal_jitter_frames: int = 0,
                 drop_invalid: bool = True,
                 ) -> None:
        super().__init__()
        self.npz_path = npz_path
        self.split = split
        self.use_norm = use_norm
        self.temporal_jitter_frames = int(temporal_jitter_frames)
        self.drop_invalid = drop_invalid
        d = np.load(npz_path, allow_pickle=True)
        self.feature_list = list(d['feature_list']) if 'feature_list' in d else None
        key_prefix = 'short' if split=='short' else 'long'
        self.x = d[f'{key_prefix}_norm'] if use_norm else d[f'{key_prefix}_raw']  # (N,T,F)
        # Some legacy npz don't contain mask fields; create all-True mask if missing.
        mask_key = f'{key_prefix}_mask'
        if mask_key in d:
            self.mask = d[mask_key]
        else:
            # Expect x shape (N, T, F); build True mask
            self.mask = np.ones_like(self.x, dtype=bool)
        any_key = f'{key_prefix}_mask_any'
        self.mask_any = d[any_key] if any_key in d else None  # (N,T) bool (optional)
        self.y = d[f'{key_prefix}_label']
        self.weight = d[f'{key_prefix}_weight'] if f'{key_prefix}_weight' in d else np.ones((self.x.shape[0],), dtype=np.float32)
        # meta
        meta_keys = [f'{key_prefix}_video_id', f'{key_prefix}_start_frame', f'{key_prefix}_end_frame']
        self.meta = {}
        for k in meta_keys:
            if k in d:
                self.meta[k] = d[k]
        # sanity: drop rows with all-invalid mask if requested
        if drop_invalid:
            valid = np.any(self.mask, axis=(1,2))
            self.keep_idx = np.where(valid)[0]
        else:
            self.keep_idx = np.arange(self.x.shape[0])

    def __len__(self) -> int:
        return len(self.keep_idx)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        T = _ensure_torch()
        i = self.keep_idx[idx]
        x = self.x[i]          # (T,F)
        m = self.mask[i]       # (T,F)
        y = int(self.y[i])
        w = float(self.weight[i]) if np.ndim(self.weight)>0 else float(self.weight)
        # temporal jitter by roll
        if self.temporal_jitter_frames:
            k = int(self.temporal_jitter_frames)
            shift = int(np.random.randint(-k, k+1))
            if shift != 0:
                x = np.roll(x, shift, axis=0)
                m = np.roll(m, shift, axis=0)
        out = {
            'x': T.from_numpy(x.astype(np.float32)),
            'mask': T.from_numpy(m.astype(np.bool_)),
            'y': T.tensor(y, dtype=T.long),
            'weight': T.tensor(w, dtype=T.float32),
            'meta': {k: (self.meta[k][i] if k in self.meta else None) for k in self.meta}
        }
        return out


def build_sampler(dataset: WindowsNPZDataset,
                  balance_by_class: bool = True,
                  amplify_hard_negative: bool = True,
                  hard_negative_factor: float = 1.0):
    T = _ensure_torch()
    N = len(dataset)
    if not balance_by_class and not amplify_hard_negative:
        return None
    # gather labels and weights
    ys = np.array([int(dataset.y[dataset.keep_idx[i]]) for i in range(N)])
    ws = np.array([float(dataset.weight[dataset.keep_idx[i]]) for i in range(N)], dtype=np.float32)
    # class balance inverse frequency
    cw = np.ones_like(ws, dtype=np.float32)
    if balance_by_class:
        pos = (ys==1).sum(); neg = (ys==0).sum()
        if pos>0 and neg>0:
            w_pos = neg / (pos + neg)
            w_neg = pos / (pos + neg)
            cw = np.where(ys==1, w_pos, w_neg).astype(np.float32)
    # hard negative amplify: assume hard negatives have sample weight < 1
    if amplify_hard_negative:
        hn_mask = (ws < 1.0)
        cw = cw * np.where(hn_mask, float(hard_negative_factor), 1.0)
    final_w = cw * ws
    import importlib
    torch_data = importlib.import_module('torch.utils.data')
    WeightedRandomSampler = getattr(torch_data, 'WeightedRandomSampler')
    sampler = WeightedRandomSampler(T.from_numpy(final_w), num_samples=N, replacement=True)
    return sampler


def build_dataloader(dataset: WindowsNPZDataset,
                     batch_size: int = 64,
                     num_workers: int = 0,
                     balance_by_class: bool = True,
                     amplify_hard_negative: bool = True,
                     hard_negative_factor: float = 1.0,
                     ):
    import importlib
    torch_data = importlib.import_module('torch.utils.data')
    sampler = build_sampler(dataset, balance_by_class, amplify_hard_negative, hard_negative_factor)
    dl = torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl
