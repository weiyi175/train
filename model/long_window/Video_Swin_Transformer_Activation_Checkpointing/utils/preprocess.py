#!/usr/bin/env python3
"""前處理工具: 將單一或多個 (T, F=36) 序列轉為模型需要之張量 (B, T, C, H, W).

步驟:
1. reshape 每時間步 36 -> (H,W)  (預設 6x6)
2. unsqueeze channel -> (1,H,W)
3. repeat channels -> (C,H,W)
4. 堆疊 batch -> (B,T,C,H,W)
"""
from __future__ import annotations
from typing import Sequence, Union, Tuple
import torch

FeatureSeq = Union[torch.Tensor, 'np.ndarray']  # type: ignore


def to_pseudo_video(seqs: Sequence[FeatureSeq], feature_grid: Tuple[int,int]=(6,6), replicate_channels: int=3) -> torch.Tensor:
    import numpy as np
    H,W = feature_grid
    out = []
    for s in seqs:
        if isinstance(s, np.ndarray):
            t = torch.from_numpy(s).float()
        else:
            t = s.float()
        assert t.ndim == 2, "Input sequence must be (T,F)"
        T,F = t.shape
        assert F == H*W, f"Feature dim {F} != H*W {H*W}"
        t = t.view(T,H,W).unsqueeze(1)  # (T,1,H,W)
        if replicate_channels > 1:
            t = t.repeat(1,replicate_channels,1,1)
        out.append(t)  # (T,C,H,W)
    # pad variable lengths? 這裡假設同長 (若不同可後續補齊)
    maxT = max(o.shape[0] for o in out)
    C = out[0].shape[1]
    padded = []
    for o in out:
        if o.shape[0] < maxT:
            pad_len = maxT - o.shape[0]
            pad_frame = torch.zeros(1,C,H,W, device=o.device, dtype=o.dtype)
            o = torch.cat([o, pad_frame.repeat(pad_len,1,1,1)], dim=0)
        padded.append(o)
    batch = torch.stack(padded,0)  # (B,T,C,H,W)
    return batch

__all__ = ["to_pseudo_video"]
