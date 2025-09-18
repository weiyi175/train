import torch
import torch.nn as nn
from typing import Sequence


class StatPoolMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (512, 128), dropout: float = 0.5, num_classes: int = 1, single_logit: bool = True, use_bn: bool = False):
        """
        Configurable MLP for StatPool features.
        - hidden_sizes: sequence of hidden layer sizes (applied in order)
        - use_bn: whether to insert BatchNorm1d after each Linear
        """
        super().__init__()
        self.single_logit = single_logit
        out_dim = 1 if single_logit else num_classes

        layers = []
        in_dim = input_dim
        # initial dropout
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        # final projection
        layers.append(nn.Linear(in_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, D)
        return self.net(x)
