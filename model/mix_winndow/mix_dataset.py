from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Literal, Tuple

class OfflineFusionDataset(Dataset):
    """Dataset for offline fusion of short/long window model outputs.

    Inputs are probabilities or logits for the positive class from two models.
    Assumes arrays are already aligned (same ordering, length N).
    If logits provided, set from_logits=True to apply sigmoid before usage.
    """
    def __init__(self, short_path: str, long_path: str, labels_path: str,
                 from_logits: bool = False, eps: float = 1e-6,
                 transform: Literal['none','logit_clip'] = 'none'):
        self.short = np.load(short_path).astype('float32')  # (N,)
        self.long = np.load(long_path).astype('float32')    # (N,)
        self.labels = np.load(labels_path).astype('int64')  # (N,)
        assert self.short.shape == self.long.shape == self.labels.shape, 'Shape mismatch between inputs and labels'
        if from_logits:
            # convert logits -> probs via sigmoid; if values already probs this is harmless if in [0,1]
            self.short = 1/(1+np.exp(-self.short))
            self.long = 1/(1+np.exp(-self.long))
        if transform == 'logit_clip':
            def _clip01(p):
                return np.clip(p, eps, 1-eps)
            self.short = _clip01(self.short)
            self.long = _clip01(self.long)
        self.N = self.short.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        p_s = float(self.short[idx])
        p_l = float(self.long[idx])
        y = int(self.labels[idx])
        return {'p_short': p_s, 'p_long': p_l, 'y': y}

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.short.copy(), self.long.copy(), self.labels.copy()
