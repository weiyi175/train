from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Simple learned-adjacency graph conv over feature nodes.
    Input: x (B, T, F)
    Output: (B, T, F_out)
    """
    def __init__(self, in_feats: int, out_feats: int):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        # linear transform per node
        self.lin = nn.Linear(in_feats, out_feats, bias=True)
        # learned adjacency (F x F)
        self.A = nn.Parameter(torch.eye(in_feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        # flat nodes: (B*T, F)
        x_flat = x.reshape(-1, F)
        # adjacency message: (B*T, F)
        msg = (self.A @ x_flat.T).T
        # combine and project
        out = self.lin(x_flat + msg)
        out = out.reshape(B, T, self.out_feats)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        y = self.dropout(y)
        return y


class GCN_TCN_Classifier(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2, gcn_hidden: int = 64,
                 tcn_channels: Tuple[int, ...] = (128, 128), tcn_kernel: int = 3, tcn_dropout: float = 0.1,
                 tcn_dil_growth: int = 2, fc_hidden: int = 128, fc_dropout: float = 0.2):
        super().__init__()
        self.gcn = GraphConv(in_dim, gcn_hidden)
        # TCN stack
        chs = [gcn_hidden] + list(tcn_channels)
        blocks = []
        for i in range(len(tcn_channels)):
            in_ch = chs[i]
            out_ch = chs[i+1]
            dilation = tcn_dil_growth ** i
            blocks.append(TemporalBlock(in_ch, out_ch, kernel_size=tcn_kernel, dilation=dilation, dropout=tcn_dropout))
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Sequential(
            nn.Linear(chs[-1], fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, n_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, T, F)
        B, T, F = x.shape
        h = self.gcn(x)  # (B, T, C)
        # prepare for TCN: (B, C, T)
        h = h.permute(0, 2, 1)
        h = self.tcn(h)
        # back to (B, T, C)
        h = h.permute(0, 2, 1)
        # masked global average pooling over time
        if mask is None:
            pooled = h.mean(dim=1)
        else:
            m = mask.float().unsqueeze(-1)  # (B, T, 1)
            s = m.sum(dim=1).clamp(min=1.0)
            pooled = (h * m).sum(dim=1) / s
        logits = self.fc(pooled)
        return logits, pooled
