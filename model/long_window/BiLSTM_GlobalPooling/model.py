import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim // 2 if dim>=2 else dim),
            nn.Tanh(),
            nn.Linear(dim // 2 if dim>=2 else dim, 1)
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # x: (B, T, D)
        scores = self.att(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        out = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return out


class BiLSTMGlobalPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.3,
                 bidirectional: bool = True, pooling: str = 'avg', num_classes: int = 1, use_bn: bool = False,
                 single_logit: bool = True):
        super().__init__()
        self.single_logit = single_logit
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers>1 else 0.0, bidirectional=bidirectional)

        pooled_dim = hidden_size * self.num_directions
        if pooling == 'attn':
            self.pool = AttentionPooling(pooled_dim)
        else:
            self.pool = None

        self.use_bn = use_bn
        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(pooled_dim))
        # final classifier
        out_dim = 1 if single_logit else num_classes
        layers.append(nn.Linear(pooled_dim, out_dim))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x, lengths=None, mask=None):
        # x: (B, T, F)
        # lengths: optional (B,) cpu tensor
        if lengths is not None:
            # pack padded sequence
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        # pooling over time
        if self.pooling == 'avg':
            if mask is None:
                res = out.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
                res = (out * mask.unsqueeze(-1)).sum(dim=1) / denom
        elif self.pooling == 'max':
            if mask is None:
                res, _ = out.max(dim=1)
            else:
                neg_inf = torch.finfo(out.dtype).min
                out2 = out.masked_fill(~mask.unsqueeze(-1), neg_inf)
                res, _ = out2.max(dim=1)
        elif self.pooling == 'attn':
            res = self.pool(out, mask)
        else:
            # default to avg
            if mask is None:
                res = out.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
                res = (out * mask.unsqueeze(-1)).sum(dim=1) / denom

        if self.use_bn:
            res = self.classifier[0](res)
            logits = self.classifier[1](res)
        else:
            logits = self.classifier(res)
        return logits
