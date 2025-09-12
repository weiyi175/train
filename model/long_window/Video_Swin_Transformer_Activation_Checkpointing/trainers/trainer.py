#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path
import sys

# 保證 utils 可被找到
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from utils.metrics import compute_all_metrics  # type: ignore


class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler, scaler, criterion, cfg, logger):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.criterion = criterion
        self.cfg = cfg
        self.logger = logger
        self.accum_steps = cfg['data']['accumulation_steps']
        self.print_freq = cfg['log']['print_freq']
        self.global_step = 0

    def train_epoch(self, loader, epoch: int):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_n = 0
        for step, batch in enumerate(loader):
            frames = batch['frames'].to(torch.float32).to(self.cfg['device'])  # (B,T,C,H,W)
            labels = batch['label'].to(self.cfg['device'])
            with torch.autocast(device_type=('cuda' if self.cfg['device']=='cuda' else 'cpu'),
                                dtype=torch.float16, enabled=self.cfg['training']['use_amp'] and self.cfg['device']=='cuda'):
                logits, _ = self.model(frames.permute(0,1,2,3,4))  # already correct shape
                loss = self.criterion(logits, labels) / self.accum_steps
            if self.cfg['training']['use_amp'] and self.cfg['device']=='cuda':
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if (step + 1) % self.accum_steps == 0:
                if self.cfg['training']['grad_clip'] > 0:
                    if self.cfg['training']['use_amp'] and self.cfg['device']=='cuda':
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg['training']['grad_clip'])
                if self.cfg['training']['use_amp'] and self.cfg['device']=='cuda':
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                batch_loss = loss.item() * self.accum_steps
                total_loss += batch_loss * labels.size(0)
                preds = logits.argmax(-1)
                total_correct += (preds == labels).sum().item()
                total_n += labels.size(0)
            self.global_step += 1
            if (step+1) % self.print_freq == 0:
                self.logger.info(f"Epoch {epoch} Step {step+1}/{len(loader)} loss={batch_loss:.4f}")
        avg_loss = total_loss / max(1,total_n)
        acc = total_correct / max(1,total_n)
        return {'train_loss': avg_loss, 'train_acc': acc}

    @torch.no_grad()
    def validate(self, loader, epoch: int):
        self.model.eval()
        total_loss = 0.0; logits_all=[]; labels_all=[]
        for batch in loader:
            frames = batch['frames'].to(torch.float32).to(self.cfg['device'])
            labels = batch['label'].to(self.cfg['device'])
            with torch.no_grad():
                logits, _ = self.model(frames.permute(0,1,2,3,4))
                loss = self.criterion(logits, labels)
            total_loss += loss.item()*labels.size(0)
            logits_all.append(logits.detach())
            labels_all.append(labels.detach())
        logits_cat = torch.cat(logits_all, 0)
        labels_cat = torch.cat(labels_all, 0)
        metrics = compute_all_metrics(logits_cat, labels_cat)
        # 混淆矩陣 (binary 假設 label {0,1})
        preds = logits_cat.argmax(-1)
        tn = ((preds==0) & (labels_cat==0)).sum().item()
        tp = ((preds==1) & (labels_cat==1)).sum().item()
        fp = ((preds==1) & (labels_cat==0)).sum().item()
        fn = ((preds==0) & (labels_cat==1)).sum().item()
        val_loss = total_loss / max(1, labels_cat.size(0))
        return {
            'val_loss': val_loss,
            'val_acc': metrics['acc'],
            'val_precision': metrics['precision'],
            'val_recall': metrics['recall'],
            'val_f1': metrics['f1'],
            'val_auc': metrics['auc'],
            'val_tn': tn,
            'val_fp': fp,
            'val_fn': fn,
            'val_tp': tp,
        }

    def step_scheduler(self, metrics: Dict[str,float]):
        if self.scheduler is None: return
        if hasattr(self.scheduler, 'step'):  # plateau needs val_loss
            try:
                self.scheduler.step(metrics.get('val_loss', None))
            except TypeError:
                self.scheduler.step()
