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
try:
    from feature_modules.feature_pack import FeaturePackBuilder, FeaturePackConfig  # type: ignore
except Exception:  # graceful degradation if not present
    FeaturePackBuilder = None  # type: ignore
    FeaturePackConfig = None  # type: ignore


class SmokeLongWindowPseudoVideo(Dataset):
    def __init__(self, npz_path: str, split: str = 'long', use_norm: bool = True, temporal_jitter: int = 0,
                 feature_grid=(6, 6), replicate_channels: int = 3,
                 use_feature_pack: bool = False,
                 fp_velocity: bool = True, fp_accel: bool = True, fp_energy: bool = True, fp_pairwise: bool = True,
                 fp_joints: int = 15, fp_dims_per_joint: int = 4, fp_pairwise_subset: int = 20):
        self.base = WindowsNPZDataset(npz_path=npz_path, split=split, use_norm=use_norm,
                                      temporal_jitter_frames=temporal_jitter)
        self.use_feature_pack = bool(use_feature_pack and FeaturePackBuilder is not None)
        self.H, self.W = feature_grid
        self.replicate_channels = replicate_channels
        self.builder = None
        if self.use_feature_pack:
            # 建立 FeaturePackConfig
            cfg = FeaturePackConfig(
                use_velocity=fp_velocity,
                use_accel=fp_accel,
                use_energy=fp_energy,
                use_pairwise=fp_pairwise,
                joints=fp_joints,
                dims_per_joint=fp_dims_per_joint,
                pairwise_subset=fp_pairwise_subset
            ) if FeaturePackConfig else None
            if cfg and FeaturePackBuilder:
                self.builder = FeaturePackBuilder(cfg)
        # 基礎形狀檢查：只有在未啟用 feature pack 時才強制 F == H*W
        sample_shape = self.base[0]['x'].shape  # could be (T,F) or (F,T)
        feat_dim_candidate = sample_shape[1] if sample_shape[1] <= sample_shape[0] else sample_shape[0]
        if not self.use_feature_pack:
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
        if torch.isnan(x_t).any():
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        # 將資料轉為 (T,F)
        if x_t.ndim != 2:
            x_t = x_t.view(x_t.shape[0], -1)
        if x_t.shape[0] == self.H * self.W and x_t.shape[1] != self.H * self.W:
            x_t = x_t.t()
        elif x_t.shape[1] == self.H * self.W:
            pass
        elif not self.use_feature_pack:
            # 只有在非 feature pack 模式才強制
            if x_t.shape[0] < x_t.shape[1]:
                x_t = x_t
            else:
                x_t = x_t.t()
        # Feature pack 模式：建立多組特徵後 pack 成 (T,C,H,W)
        if self.use_feature_pack and self.builder is not None:
            feat_dict = self.builder.build(x_t)  # 各 component: (T, Fk)
            x_t = self.builder.pack_for_model(feat_dict, (self.H, self.W))  # (T,C,H,W)
        else:
            T, F = x_t.shape
            x_t = x_t.view(T, self.H, self.W).unsqueeze(1)  # (T,1,H,W)
            if self.replicate_channels > 1:
                x_t = x_t.repeat(1, self.replicate_channels, 1, 1)
        y = int(item['y'])
        w = float(item.get('weight', 1.0))
        return {
            'frames': x_t,   # (T,C,H,W)
            'label': y,
            'weight': w,
        }


def build_dataloaders(npz_path: str, batch_size_micro: int, val_ratio: float, seed: int, num_workers: int,
                      balance_by_class: bool, amplify_hard_negative: bool, hard_negative_factor: float,
                      temporal_jitter: int, feature_grid, replicate_channels: int, use_sampler: bool = True,
                      use_feature_pack: bool = False,
                      fp_velocity: bool = True, fp_accel: bool = True, fp_energy: bool = True, fp_pairwise: bool = True,
                      fp_joints: int = 15, fp_dims_per_joint: int = 4, fp_pairwise_subset: int = 20):
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    ds = SmokeLongWindowPseudoVideo(
        npz_path=npz_path, split='long', use_norm=True, temporal_jitter=temporal_jitter,
        feature_grid=feature_grid, replicate_channels=replicate_channels,
        use_feature_pack=use_feature_pack,
        fp_velocity=fp_velocity, fp_accel=fp_accel, fp_energy=fp_energy, fp_pairwise=fp_pairwise,
        fp_joints=fp_joints, fp_dims_per_joint=fp_dims_per_joint, fp_pairwise_subset=fp_pairwise_subset
    )
    N = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    n_val = max(1, int(val_ratio * N))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)
    # class balancing weights (optional)
    import numpy as np
    ys = np.array([ds[i]['label'] for i in train_idx], dtype=np.int64)
    base_w = np.array([float(ds[i]['weight']) for i in train_idx], dtype=np.float32)
    sampler = None
    if use_sampler and (balance_by_class or amplify_hard_negative):
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

    train_loader = DataLoader(train_ds, batch_size=batch_size_micro, shuffle=(sampler is None), sampler=sampler,
                              num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size_micro, shuffle=False,
                            num_workers=num_workers, pin_memory=True, collate_fn=collate)
    sample = ds[0]['frames']
    class_counts = np.bincount(ys, minlength=2)
    meta = {
        'T': sample.shape[0], 'C': sample.shape[1], 'H': sample.shape[2], 'W': sample.shape[3], 'N': N,
        'train_class_counts': class_counts,
        'use_feature_pack': use_feature_pack,
        'feature_pack_components': {
            'velocity': fp_velocity, 'accel': fp_accel, 'energy': fp_energy, 'pairwise': fp_pairwise
        } if use_feature_pack else None,
        'feature_pack_joints': fp_joints if use_feature_pack else None,
        'feature_pack_dims_per_joint': fp_dims_per_joint if use_feature_pack else None,
        'feature_pack_pairwise_subset': fp_pairwise_subset if use_feature_pack else None
    }
    return train_loader, val_loader, meta

__all__ = ['SmokeLongWindowPseudoVideo', 'build_dataloaders']
