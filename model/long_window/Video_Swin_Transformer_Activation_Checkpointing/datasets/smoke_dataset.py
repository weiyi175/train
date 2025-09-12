#!/usr/bin/env python3
from __future__ import annotations
"""Dataset wrapper: 將現有 WindowsNPZDataset (long) 轉為 pseudo video tensor。

輸入原始: (T, F=36)
轉換: 每個時間步 36 -> 6x6 grid -> (1,6,6) -> repeat 3 channel -> (3,6,6)
產出張量: (T, 3, 6, 6)

模型需要 (B, T, H, W, C)，train loop 會再 permute。
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

# 本檔案位置: .../model/long_window/Video_Swin_Transformer_Activation_Checkpointing/datasets/smoke_dataset.py
ROOT = Path(__file__).resolve().parents[4]  # 回到 train 專案根目錄 (/home/user/projects/train)
TOOL = ROOT / 'Tool'
if str(TOOL) not in sys.path:
    sys.path.append(str(TOOL))
from dataset_npz import WindowsNPZDataset  # type: ignore


class SmokeLongWindowPseudoVideo(Dataset):
    def __init__(self, npz_path: str, split: str = 'long', use_norm: bool = True, temporal_jitter: int = 0,
                 feature_grid=(6, 6), replicate_channels: int = 3):
        self.base = WindowsNPZDataset(npz_path=npz_path, split=split, use_norm=use_norm,
                                      temporal_jitter_frames=temporal_jitter)
        self.H, self.W = feature_grid
        self.C = replicate_channels
        sample_shape = self.base[0]['x'].shape  # could be (T,F) or (F,T)
        feat_dim_candidate = sample_shape[1] if sample_shape[1] <= sample_shape[0] else sample_shape[0]
        assert self.H * self.W == feat_dim_candidate, (
            f"feature_grid 面積需等於特徵維度數: grid={self.H*self.W} sample_shape={sample_shape}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        x = item['x']  # expected (T, F) but source may be (F, T)
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
        else:
            x_t = x.float()
        # Normalize orientation: we choose smaller dimension as feature dim if ambiguous.
        if x_t.shape[0] == self.H * self.W and x_t.shape[1] != self.H * self.W:
            x_t = x_t.t()  # (F,T)->(T,F)
        elif x_t.shape[1] == self.H * self.W:
            pass  # already (T,F)
        elif x_t.shape[0] == self.H * self.W and x_t.shape[1] == self.H * self.W:
            # square; assume time on axis0
            pass
        else:
            # fallback: if second dim larger, transpose
            if x_t.shape[0] < x_t.shape[1]:
                x_t = x_t  # (T,F) already
            else:
                x_t = x_t.t()
        T, F = x_t.shape
        x_t = x_t.view(T, self.H, self.W).unsqueeze(1)  # (T,1,H,W)
        if self.C > 1:
            x_t = x_t.repeat(1, self.C, 1, 1)  # (T,C,H,W)
        y = int(item['y'])
        w = float(item.get('weight', 1.0))
        return {
            'frames': x_t,   # (T,C,H,W)
            'label': y,
            'weight': w,
        }


def build_dataloaders(npz_path: str, batch_size_micro: int, val_ratio: float, seed: int, num_workers: int,
                      balance_by_class: bool, amplify_hard_negative: bool, hard_negative_factor: float,
                      temporal_jitter: int, feature_grid, replicate_channels: int):
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    ds = SmokeLongWindowPseudoVideo(npz_path=npz_path, split='long', use_norm=True,
                                    temporal_jitter=temporal_jitter, feature_grid=feature_grid,
                                    replicate_channels=replicate_channels)
    N = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    n_val = max(1, int(val_ratio * N))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)
    # class balancing weights
    import numpy as np
    ys = np.array([ds[i]['label'] for i in train_idx], dtype=np.int64)
    base_w = np.array([float(ds[i]['weight']) for i in train_idx], dtype=np.float32)
    sw = np.ones_like(base_w, dtype=np.float32)
    if balance_by_class:
        cnt = np.bincount(ys, minlength=2).astype(np.float32)
        cw = 1.0 / np.maximum(cnt, 1.0)
        cw = cw / cw.sum() * 2.0
        sw *= cw[ys]
    if amplify_hard_negative:
        hn = base_w < 1.0
        sw[hn] *= float(hard_negative_factor)
    sampler = WeightedRandomSampler(weights=torch.tensor(sw), num_samples=len(sw), replacement=True)

    def collate(batch):
        frames = torch.stack([b['frames'] for b in batch], 0)  # (B,T,C,H,W)
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        weights = torch.tensor([b['weight'] for b in batch], dtype=torch.float32)
        return {'frames': frames, 'label': labels, 'weight': weights}

    train_loader = DataLoader(train_ds, batch_size=batch_size_micro, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size_micro, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate)
    sample = ds[0]['frames']
    meta = {'T': sample.shape[0], 'C': sample.shape[1], 'H': sample.shape[2], 'W': sample.shape[3], 'N': N}
    return train_loader, val_loader, meta

__all__ = ['SmokeLongWindowPseudoVideo', 'build_dataloaders']
