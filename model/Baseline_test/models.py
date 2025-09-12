from __future__ import annotations
import math
import torch
import torch.nn as nn


class MLPFlat(nn.Module):
    def __init__(self, T: int, F: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = T * F
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, mask=None):
        # x: (B, T, F)
        B, T, F = x.shape
        z = x.reshape(B, T * F)
        return self.net(z)


class StatPoolMLP(nn.Module):
    def __init__(self, F: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = F * 2  # mean + std
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, mask=None):
        # x: (B, T, F); mask: (B, T, F) bool
        if mask is not None:
            # masked mean/std across time per feature
            m = mask.float()
            denom = m.sum(dim=1).clamp(min=1.0)  # (B, F)
            xm = (x * m).sum(dim=1) / denom
            var = ((x - xm.unsqueeze(1))**2 * m).sum(dim=1) / denom
            xs = var.clamp(min=1e-8).sqrt()
        else:
            xm = x.mean(dim=1)
            xs = x.std(dim=1)
        feat = torch.cat([xm, xs], dim=-1)
        return self.net(feat)
