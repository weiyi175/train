from __future__ import annotations
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """單層或雙層 LSTM，支援可選 bidirectional，輸入 (B,T,F)。
    取最後時間步的隱狀態（或 concat 雙向）後接 MLP。"""
    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1,
                 bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=(dropout if num_layers > 1 else 0.0),
                            bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, 2)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        # x: (B, T, F)
        # 可選 mask：這裡簡化，直接餵入 LSTM；NaN 已在 dataloader 前處理為 0
        y, (hn, cn) = self.lstm(x)
        if self.lstm.bidirectional:
            # 取最後層的雙向 hidden concat
            last_h = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            last_h = hn[-1]
        z = self.norm(last_h)
        logits = self.head(z)
        return logits
