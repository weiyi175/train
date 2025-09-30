from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedAvgFusion(nn.Module):
    """Learnable weighted average of two probability inputs.
    Forward expects tensor (B,2) = [p_short, p_long].
    Output logits for 2 classes (binary)."""

    def __init__(self):
        super().__init__()
        # raw weights -> softmax to ensure they form a convex combination
        self.raw_w = nn.Parameter(torch.zeros(2))

    def forward(self, p: torch.Tensor):
        # p: (B,2) with values in [0,1]
        w = torch.softmax(self.raw_w, dim=0)  # (2,)
        ps = p * w  # (B,2)
        fused = ps.sum(dim=1)  # (B,)
        fused = fused.clamp(1e-6, 1 - 1e-6)
        # convert prob to logits for 2 classes, shape (B,2)
        logit_pos = torch.log(fused / (1 - fused)).unsqueeze(-1)  # (B,1)
        logits = torch.cat([torch.zeros_like(logit_pos), logit_pos], dim=-1)  # (B,2)
        return logits

class MLPFusion(nn.Module):
    """MLP on feature expansion: [p_s, p_l, p_s*p_l, |p_s - p_l|]."""
    def __init__(self, hidden: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 2)
        )
    def forward(self, p: torch.Tensor):
        # p: (B,2)
        p_s = p[:,0:1]; p_l = p[:,1:2]
        feats = torch.cat([p_s, p_l, p_s*p_l, (p_s - p_l).abs()], dim=1)
        return self.net(feats)

class LogisticStackFusion(nn.Module):
    """Simple logistic regression on [p_short, p_long]."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2,1)
    def forward(self, p: torch.Tensor):
        z = self.lin(p)  # (B,1)
        logits = torch.cat([torch.zeros_like(z), z], dim=-1)
        return logits

FUSION_BUILDERS = {
    'weighted': lambda: WeightedAvgFusion(),
    'mlp': lambda: MLPFusion(),
    'stack_logistic': lambda: LogisticStackFusion(),
}
