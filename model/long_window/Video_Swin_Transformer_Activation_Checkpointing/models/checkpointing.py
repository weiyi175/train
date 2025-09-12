#!/usr/bin/env python3
from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def apply_activation_checkpointing(model: nn.Module, layer_prefixes):
    """Wrap forward of modules whose full name starts with any prefix in layer_prefixes."""
    for name, module in list(model.named_modules()):
        if any(name.startswith(p) for p in layer_prefixes):
            # skip root model
            if len(list(module.children())) == 0:
                continue
            # wrap only block modules having a forward
            if hasattr(module, 'forward'):
                orig_forward = module.forward
                def wrapped_forward(*args, _orig=orig_forward, **kwargs):
                    return checkpoint(_orig, *args, **kwargs)
                module.forward = wrapped_forward  # type: ignore
    return model

__all__ = ['apply_activation_checkpointing']
