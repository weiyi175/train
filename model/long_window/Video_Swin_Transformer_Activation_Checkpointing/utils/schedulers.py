#!/usr/bin/env python3
from __future__ import annotations
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch

class WarmupCosine(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        epoch = self.last_epoch + 1
        lrs = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch / max(1,self.warmup_epochs))
            else:
                progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5*(base_lr - self.min_lr)*(1+math.cos(math.pi*progress))
            lrs.append(lr)
        return lrs

def build_scheduler(optimizer, cfg):
    sch_cfg = cfg['scheduler']
    train_cfg = cfg['training']
    if sch_cfg['type'] == 'cosine':
        return WarmupCosine(optimizer, warmup_epochs=sch_cfg['warmup_epochs'], max_epochs=train_cfg['epochs'], min_lr=sch_cfg['min_lr'])
    return None

__all__ = ['build_scheduler', 'WarmupCosine']
