#!/usr/bin/env python3
from __future__ import annotations
"""Training script for Video Swin 3D feature model (pseudo video smoke dataset).

Features:
- F1 / AUC / Confusion Matrix metrics with early stopping on F1.
- Gradient accumulation (micro-batch) & AMP.
- Checkpoint saving (best + last) with config.
- Optional activation checkpointing (stage >= 2) via model flag.

Usage (example):
    python scripts/train_videoswin.py \
        --npz_path /path/to/data.npz \
        --preset tiny \
        --epochs 30 --batch 8 --accum 2 --lr 3e-4 \
        --out runs/swin_tiny

"""
import argparse, json
from pathlib import Path
import sys, math, time
import torch
import torch.nn.functional as F
from torch import nn

try:
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix  # type: ignore
except Exception:  # fallback minimal implementations
    import numpy as _np
    def f1_score(y_true, y_pred, average='binary'):
        tp = ((y_true==1)&(y_pred==1)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        precision = tp / max(1, tp+fp)
        recall = tp / max(1, tp+fn)
        return 2*precision*recall/max(1e-8, precision+recall)
    def roc_auc_score(y_true, y_prob):
        # simple rank-based AUC (binary)
        order = _np.argsort(-y_prob)
        y = y_true[order]
        pos = (y==1)
        n_pos = pos.sum(); n_neg = len(y)-n_pos
        if n_pos==0 or n_neg==0:
            raise ValueError('AUC undefined')
        cum_pos = _np.cumsum(pos)
        rank_pos_sum = cum_pos[pos].sum()
        return (rank_pos_sum - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    def confusion_matrix(y_true, y_pred):
        tp = ((y_true==1)&(y_pred==1)).sum()
        tn = ((y_true==0)&(y_pred==0)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        return _np.array([[tn, fp],[fn, tp]])


# Ensure base path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.videoswin import build_videoswin3d_preset, build_videoswin3d_feature, VideoSwin3DConfig
from datasets.smoke_dataset import build_dataloaders


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_path', required=True)
    ap.add_argument('--preset', default='tiny', help='tiny|small|base or custom if --depths provided')
    ap.add_argument('--depths', type=int, nargs='*', help='Override depths list')
    ap.add_argument('--num_heads', type=int, nargs='*', help='Override num_heads list')
    ap.add_argument('--embed_dim', type=int, help='Override embed_dim')
    ap.add_argument('--window_size', type=int, nargs=3, default=(2,7,7))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--warmup_epochs', type=int, default=2)
    ap.add_argument('--drop_path_rate', type=float, help='Override drop path')
    ap.add_argument('--use_checkpoint', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--temporal_jitter', type=int, default=0)
    ap.add_argument('--feature_grid', type=int, nargs=2, default=(6,6))
    ap.add_argument('--replicate_channels', type=int, default=3)
    ap.add_argument('--balance_by_class', action='store_true')
    ap.add_argument('--amplify_hard_negative', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=2.0)
    ap.add_argument('--early_patience', type=int, default=6)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--amp', action='store_true')
    return ap.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model(args, num_classes: int):
    overrides = {}
    if args.depths: overrides['depths'] = tuple(args.depths)
    if args.num_heads: overrides['num_heads'] = tuple(args.num_heads)
    if args.embed_dim: overrides['embed_dim'] = args.embed_dim
    if args.drop_path_rate is not None: overrides['drop_path_rate'] = args.drop_path_rate
    if overrides:
        # Build from preset then override
        model = build_videoswin3d_preset(args.preset, num_classes=num_classes, window_size=tuple(args.window_size),
                                         use_checkpoint=args.use_checkpoint, **overrides)
    else:
        model = build_videoswin3d_preset(args.preset, num_classes=num_classes, window_size=tuple(args.window_size),
                                         use_checkpoint=args.use_checkpoint)
    return model


def cosine_lr(step, total_steps, base_lr, min_lr=1e-6, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def train_one_epoch(model, loader, optimizer, scaler, device, accum, global_step, total_steps, base_lr, warmup_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        frames = batch['frames'].to(device)  # (B,T,C,H,W)
        labels = batch['label'].to(device)
        # Permute to (B,T,C,H,W) already correct from collate
        with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=scaler is not None):
            logits, _ = model(frames)
            loss = F.cross_entropy(logits, labels) / accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * accum
        step_this = global_step + i
        lr = cosine_lr(step_this, total_steps, base_lr, warmup_steps=warmup_steps)
        for pg in optimizer.param_groups: pg['lr'] = lr
        if (i + 1) % accum == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, len(loader))


def evaluate(model, loader, device):
    model.eval()
    ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(frames)
            prob = logits.softmax(-1)
            pred = prob.argmax(-1)
            ys.append(labels.cpu()); ps.append(pred.cpu()); probs.append(prob[:,1].cpu())
    import torch as _t
    y = _t.cat(ys); p = _t.cat(ps); pr = _t.cat(probs)
    f1 = f1_score(y.numpy(), p.numpy(), average='binary')
    try:
        auc = roc_auc_score(y.numpy(), pr.numpy())
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(y.numpy(), p.numpy())
    acc = (p==y).float().mean().item()
    return {'f1':f1,'auc':auc,'cm':cm.tolist(),'acc':acc}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, meta = build_dataloaders(
        npz_path=args.npz_path, batch_size_micro=args.batch, val_ratio=args.val_ratio, seed=args.seed,
        num_workers=args.num_workers, balance_by_class=args.balance_by_class, amplify_hard_negative=args.amplify_hard_negative,
        hard_negative_factor=args.hard_negative_factor, temporal_jitter=args.temporal_jitter, feature_grid=tuple(args.feature_grid),
        replicate_channels=args.replicate_channels)

    num_classes = 2  # binary
    model = build_model(args, num_classes).to(device)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.startswith('cuda') else None

    best_f1 = -1.0; best_epoch=-1; patience=0
    history = []

    config_dump = {
        'data': {
            'npz_path': args.npz_path,
            'feature_grid': list(args.feature_grid),
            'replicate_channels': args.replicate_channels,
        },
        'model': {
            'preset': args.preset,
            'depths': list(model.stages[0]['blocks'].__len__() for _ in [0]),  # placeholder
        },
        'training': {
            'epochs': args.epochs,
            'batch': args.batch,
            'accum': args.accum,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }
    }

    for epoch in range(1, args.epochs+1):
        start=time.time()
        loss=train_one_epoch(model, train_loader, optimizer, scaler, device, args.accum, (epoch-1)*len(train_loader), total_steps, args.lr, warmup_steps)
        metrics=evaluate(model, val_loader, device)
        history.append({'epoch':epoch,'loss':loss,**metrics})
        improved = metrics['f1'] > best_f1 + args.min_delta
        if improved:
            best_f1 = metrics['f1']; best_epoch=epoch; patience=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics}, out/'best.ckpt')
        else:
            patience+=1
        torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics}, out/'last.ckpt')
        with (out/'history.json').open('w') as f:
            json.dump(history,f,indent=2)
        print(f"Epoch {epoch} loss={loss:.4f} f1={metrics['f1']:.4f} auc={metrics['auc']:.4f} acc={metrics['acc']:.4f} best_f1={best_f1:.4f} ({best_epoch}) time={time.time()-start:.1f}s")
        if patience >= args.early_patience:
            print('[EARLY STOP]')
            break

    print('Training complete. Best epoch', best_epoch, 'F1', best_f1)

if __name__=='__main__':
    main()
