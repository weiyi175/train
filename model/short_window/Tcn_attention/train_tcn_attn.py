#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# add Tool to path for dataset
ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'Tool'
if str(TOOL) not in sys.path:
    sys.path.append(str(TOOL))
from dataset_npz import WindowsNPZDataset  # type: ignore

# local modules
from models import TCNWithAttentionClassifier
from utils import get_next_run_dir, save_json, count_params, generate_run_report


def build_dataloaders(npz_path: str, use_norm: bool, batch_size: int, val_ratio: float, num_workers: int,
                      seed: int, balance_by_class: bool, amplify_hard_negative: bool, hard_negative_factor: float,
                      temporal_jitter_frames: int) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    ds = WindowsNPZDataset(
        npz_path=npz_path,
        split='short',
        use_norm=use_norm,
        temporal_jitter_frames=temporal_jitter_frames,
    )
    N = len(ds)
    # split
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g).tolist()
    n_val = max(1, int(N * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    # build sampler weights
    ys = np.array([ds[i]['y'] for i in train_idx], dtype=np.int64)
    base_w = np.array([float(ds[i].get('weight', 1.0)) for i in train_idx], dtype=np.float32)
    sw = np.ones_like(base_w, dtype=np.float32)
    if balance_by_class:
        cnt = np.bincount(ys, minlength=2).astype(np.float32)
        cw = 1.0 / np.maximum(cnt, 1.0)
        cw = cw / cw.sum() * 2.0
        sw *= cw[ys]
    if amplify_hard_negative:
        hn = base_w < 1.0
        sw[hn] *= float(hard_negative_factor)
    sampler = WeightedRandomSampler(weights=torch.tensor(sw), num_samples=len(sw), replacement=True)

    def collate(batch):
        xs, masks, ys, ws = [], [], [], []
        for b in batch:
            x = b['x']
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            m = b.get('mask')
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            # 將 (T,F) 的特徵遮罩轉為 (T,) 的時間遮罩
            if m is None:
                m_time = torch.ones(x.shape[0], dtype=torch.bool)
            else:
                m_bool = m.bool()
                m_time = m_bool.any(dim=-1) if m_bool.ndim == 2 else m_bool
            xs.append(torch.nan_to_num(x, nan=0.0))
            masks.append(m_time)
            ys.append(int(b['y']))
            ws.append(float(b.get('weight', 1.0)))
        X = torch.stack(xs, 0)
        M = torch.stack(masks, 0)
        Y = torch.tensor(ys, dtype=torch.long)
        W = torch.tensor(ws, dtype=torch.float32)
        return {'x': X, 'mask': M, 'y': Y, 'weight': W}

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)

    T, F = ds[0]['x'].shape
    return train_loader, val_loader, {'N': N, 'T': T, 'F': F}


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        m = batch['mask'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        logits, _ = model(x, mask=m)
        loss = criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(-1)
            n = y.size(0)
            total_loss += float(loss.item()) * n
            total_acc += float((pred == y).float().sum().item())
            total_n += n
    return total_loss / total_n, total_acc / total_n


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        m = batch['mask'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        logits, _ = model(x, mask=m)
        loss = criterion(logits, y)
        pred = logits.argmax(-1)
        n = y.size(0)
        total_loss += float(loss.item()) * n
        total_acc += float((pred == y).float().sum().item())
        total_n += n
    return total_loss / total_n, total_acc / total_n


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=str(ROOT / 'train_data' / 'slipce' / 'windows_dense_npz.npz'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--use_norm', action='store_true')
    ap.add_argument('--balance_by_class', action='store_true')
    ap.add_argument('--amplify_hard_negative', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=1.5)
    ap.add_argument('--temporal_jitter_frames', type=int, default=0)
    # model hparams
    ap.add_argument('--tcn_channels', type=str, default='128,128,128')
    ap.add_argument('--tcn_kernel', type=int, default=5)
    ap.add_argument('--tcn_dropout', type=float, default=0.1)
    ap.add_argument('--tcn_dil_growth', type=int, default=2)
    ap.add_argument('--attn_type', choices=['additive','mhsa'], default='additive')
    ap.add_argument('--attn_hidden', type=int, default=128)
    ap.add_argument('--mhsa_heads', type=int, default=4)
    ap.add_argument('--fc_hidden', type=int, default=128)
    ap.add_argument('--fc_dropout', type=float, default=0.2)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_dir = get_next_run_dir(str(Path(__file__).parent / 'result'))
    print(f"[INFO] run_dir = {run_dir}")

    tr_loader, va_loader, meta = build_dataloaders(
        npz_path=args.npz,
        use_norm=args.use_norm,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        balance_by_class=args.balance_by_class,
        amplify_hard_negative=args.amplify_hard_negative,
        hard_negative_factor=args.hard_negative_factor,
        temporal_jitter_frames=args.temporal_jitter_frames,
    )

    F = meta['F']
    model = TCNWithAttentionClassifier(
        in_dim=F,
        n_classes=2,
        tcn_channels=tuple(int(x) for x in args.tcn_channels.split(',')),
        tcn_kernel=args.tcn_kernel,
        tcn_dropout=args.tcn_dropout,
        tcn_dil_growth=args.tcn_dil_growth,
        attn_type=args.attn_type,
        attn_hidden=args.attn_hidden,
        mhsa_heads=args.mhsa_heads,
        fc_hidden=args.fc_hidden,
        fc_dropout=args.fc_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    num_params = count_params(model)
    cfg = {
        'model': 'tcn_attention', 'split': 'short', 'device': device,
        'num_params': num_params, 'epochs': args.epochs, 'batch_size': args.batch_size,
        'lr': args.lr, 'weight_decay': args.weight_decay, 'seed': args.seed, 'use_norm': args.use_norm,
        'balance_by_class': args.balance_by_class, 'amplify_hard_negative': args.amplify_hard_negative,
        'hard_negative_factor': args.hard_negative_factor, 'temporal_jitter_frames': args.temporal_jitter_frames,
        'val_ratio': args.val_ratio, 'num_workers': args.num_workers, 'params': {
            'tcn_channels': args.tcn_channels, 'tcn_kernel': args.tcn_kernel,
            'tcn_dropout': args.tcn_dropout, 'tcn_dil_growth': args.tcn_dil_growth,
            'attn_type': args.attn_type, 'attn_hidden': args.attn_hidden,
            'mhsa_heads': args.mhsa_heads, 'fc_hidden': args.fc_hidden, 'fc_dropout': args.fc_dropout,
        }, 'data': meta,
    }
    save_json(cfg, str(run_dir / 'config.json'))
    (run_dir / 'model_spec.txt').write_text(str(model), encoding='utf-8')

    best = {'val_acc': -1.0, 'epoch': -1}
    log_path = run_dir / 'train_log.jsonl'
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, device, criterion, optim)
        va_loss, va_acc = evaluate(model, va_loader, device, criterion)
        rec = {'epoch': epoch, 'train_loss': float(tr_loss), 'train_acc': float(tr_acc), 'val_loss': float(va_loss), 'val_acc': float(va_acc)}
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec)+"\n")
        if va_acc > best['val_acc']:
            best = {'val_acc': va_acc, 'epoch': epoch}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': va_acc}, run_dir / 'best.ckpt')
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': va_acc}, run_dir / 'last.ckpt')
        print(f"[E{epoch:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} | best@{best['epoch']}={best['val_acc']:.4f}")

    # post-run verification: ensure logs reached intended epochs
    actual_epochs = 0
    if log_path.exists():
        with log_path.open('r', encoding='utf-8') as f:
            for _ in f:
                actual_epochs += 1
    if actual_epochs != args.epochs:
        meta = {'intended_epochs': int(args.epochs), 'actual_logged_epochs': int(actual_epochs)}
        save_json(meta, str(run_dir / 'run_meta.json'))
        print(f"[WARN] intended epochs={args.epochs} but found only {actual_epochs} logged. Wrote run_meta.json to {run_dir}")

    generate_run_report(str(run_dir))


if __name__ == '__main__':
    # Quick-run arguments (edit here for fast tweaks). If you pass CLI args, they will override this.
    RUN_ARGS = [
        '--npz', '/home/user/projects/train/train_data/slipce/windows_dense_npz.npz',
        '--use_norm',
        '--epochs', '20',
        '--batch_size', '256',
        '--lr', '1e-4',
        '--balance_by_class',
        '--amplify_hard_negative',
        '--hard_negative_factor', '1.5',
    ]
    argv = sys.argv[1:] if len(sys.argv) > 1 else RUN_ARGS
    main(argv)
