from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = self.act(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.act(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, in_ch: int, channels: Tuple[int, ...], kernel_size: int = 5, dropout: float = 0.1, dil_growth: int = 2):
        super().__init__()
        layers = []
        ch_prev = in_ch
        dilation = 1
        for ch in channels:
            layers.append(TemporalBlock(ch_prev, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            ch_prev = ch
            dilation *= dil_growth
        self.net = nn.Sequential(*layers)
        self.out_ch = ch_prev

    def forward(self, x_btf: torch.Tensor) -> torch.Tensor:
        x = x_btf.transpose(1, 2)  # (B, F, T)
        y = self.net(x)            # (B, C, T)
        y = y.transpose(1, 2)      # (B, T, C)
        return y


class AdditiveAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = self.proj(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        pooled = torch.einsum('bt,btc->bc', attn, x)
        return pooled, attn


class MHSAEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        out, attn = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        out = self.ln(out + x)
        return out, attn


class TCNWithAttentionClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_classes: int = 2,
        tcn_channels: Tuple[int, ...] = (128, 128, 128),
        tcn_kernel: int = 5,
        tcn_dropout: float = 0.1,
        tcn_dil_growth: int = 2,
        attn_type: str = 'additive',  # 'additive' or 'mhsa'
        attn_hidden: int = 128,
        mhsa_heads: int = 4,
        fc_hidden: int = 128,
        fc_dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = TCNEncoder(in_ch=in_dim, channels=tcn_channels, kernel_size=tcn_kernel, dropout=tcn_dropout, dil_growth=tcn_dil_growth)
        enc_dim = self.encoder.out_ch
        self.attn_type = attn_type
        if attn_type == 'additive':
            self.attn_pool = AdditiveAttentionPool(enc_dim, hidden=attn_hidden)
        elif attn_type == 'mhsa':
            self.mhsa = MHSAEncoder(enc_dim, num_heads=mhsa_heads, dropout=fc_dropout)
            self.attn_pool = AdditiveAttentionPool(enc_dim, hidden=attn_hidden)
        else:
            raise ValueError(f'Unknown attn_type={attn_type}')
        self.classifier = nn.Sequential(
            nn.Linear(enc_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        h = self.encoder(x)  # (B, T, C)
        if self.attn_type == 'mhsa':
            kpm = (~mask) if mask is not None else None
            h, _ = self.mhsa(h, key_padding_mask=kpm)
        pooled, attn = self.attn_pool(h, mask=mask)
        logits = self.classifier(pooled)
        return logits, attn
