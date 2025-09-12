#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, time
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

# Reuse short-window model definition
SHORT_MODEL_DIR = Path(__file__).resolve().parents[2] / 'short_window' / 'Tcn_attention'
if str(SHORT_MODEL_DIR) not in sys.path:
    sys.path.append(str(SHORT_MODEL_DIR))
from models import TCNWithAttentionClassifier  # type: ignore
from utils import get_next_run_dir, save_json, count_params, generate_run_report  # type: ignore


def build_dataloaders(npz_path: str, use_norm: bool, batch_size: int, val_ratio: float, num_workers: int,
                      seed: int, balance_by_class: bool, amplify_hard_negative: bool, hard_negative_factor: float,
                      temporal_jitter_frames: int) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    ds = WindowsNPZDataset(
        npz_path=npz_path,
        split='long',  # 長窗
        use_norm=use_norm,
        temporal_jitter_frames=temporal_jitter_frames,
    )
    N = len(ds)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(N, generator=g).tolist()
    n_val = max(1, int(N * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

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
        xs, masks, ys_l, ws = [], [], [], []
        for b in batch:
            x = b['x']
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            m = b.get('mask')
            if isinstance(m, np.ndarray):
                m = torch.from_numpy(m)
            if m is None:
                m_time = torch.ones(x.shape[0], dtype=torch.bool)
            else:
                m_bool = m.bool()
                m_time = m_bool.any(dim=-1) if m_bool.ndim == 2 else m_bool
            xs.append(torch.nan_to_num(x, nan=0.0))
            masks.append(m_time)
            ys_l.append(int(b['y']))
            ws.append(float(b.get('weight', 1.0)))
        X = torch.stack(xs, 0)
        M = torch.stack(masks, 0)
        Y = torch.tensor(ys_l, dtype=torch.long)
        W = torch.tensor(ws, dtype=torch.float32)
        return {'x': X, 'mask': M, 'y': Y, 'weight': W}

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)

    T, F = ds[0]['x'].shape  # (75, F)
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
    ap.add_argument('--batch_size', type=int, default=128)  # 長窗時間序列更長，適度降低 batch
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--use_norm', action='store_true')
    ap.add_argument('--balance_by_class', action='store_true')
    ap.add_argument('--amplify_hard_negative', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=1.5)
    ap.add_argument('--temporal_jitter_frames', type=int, default=0)
    # model hparams (針對 T=75：增加通道與 dilation 深度來擴大感受野)
    ap.add_argument('--tcn_channels', type=str, default='128,128,256,256')
    ap.add_argument('--tcn_kernel', type=int, default=3)  # 較小 kernel 減少 padding
    ap.add_argument('--tcn_dropout', type=float, default=0.1)
    ap.add_argument('--tcn_dil_growth', type=int, default=2)
    ap.add_argument('--attn_type', choices=['additive','mhsa'], default='additive')
    ap.add_argument('--attn_hidden', type=int, default=192)
    ap.add_argument('--mhsa_heads', type=int, default=4)
    ap.add_argument('--fc_hidden', type=int, default=192)
    ap.add_argument('--fc_dropout', type=float, default=0.3)
    # regularization / optimization extras
    ap.add_argument('--label_smoothing', type=float, default=0.0)
    ap.add_argument('--lr_scheduler', choices=['none','plateau','cosine'], default='none')
    ap.add_argument('--plateau_patience', type=int, default=6)
    ap.add_argument('--plateau_factor', type=float, default=0.5)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    ap.add_argument('--cosine_tmax', type=int, default=50)
    ap.add_argument('--early_stop_patience', type=int, default=0, help='0=disable')
    ap.add_argument('--tag', type=str, default='', help='Optional run tag appended to run_dir name')
    ap.add_argument('--init_ckpt', type=str, default='', help='Optional path to a checkpoint to initialize model weights')
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_result_dir = Path(__file__).parent / 'result'
    run_dir = get_next_run_dir(str(base_result_dir))
    if args.tag:
        # rename directory with tag suffix
        tagged = run_dir.parent / f"{run_dir.name}_{args.tag}"
        if not tagged.exists():
            run_dir.rename(tagged)
            run_dir = tagged
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

    # optionally initialize from a checkpoint
    if args.init_ckpt:
        ckpt_p = Path(args.init_ckpt)
        if ckpt_p.exists():
            dd = torch.load(str(ckpt_p), map_location=device)
            if 'model' in dd:
                model.load_state_dict(dd['model'])
                print(f"[INFO] Initialized model weights from {ckpt_p}")
            else:
                print(f"[WARN] init_ckpt {ckpt_p} missing 'model' key")
        else:
            print(f"[WARN] init_ckpt {ckpt_p} not found")

    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler setup
    scheduler = None
    if args.lr_scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=args.plateau_factor,
                                                               patience=args.plateau_patience, min_lr=args.min_lr, verbose=True)
    elif args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.cosine_tmax, eta_min=args.min_lr)
    num_params = count_params(model)
    cfg = {
        'model': 'tcn_attention_long', 'split': 'long', 'device': device,
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
    best_es_epoch = 0  # for early stopping
    start_time = time.time()
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, device, criterion, optim)
        va_loss, va_acc = evaluate(model, va_loader, device, criterion)
        rec = {'epoch': epoch, 'train_loss': float(tr_loss), 'train_acc': float(tr_acc), 'val_loss': float(va_loss), 'val_acc': float(va_acc)}
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec)+"\n")
        if va_acc > best['val_acc']:
            best = {'val_acc': va_acc, 'epoch': epoch}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': va_acc}, run_dir / 'best.ckpt')
            best_es_epoch = epoch
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': va_acc}, run_dir / 'last.ckpt')
        current_lr = optim.param_groups[0]['lr']
        print(f"[E{epoch:03d}] lr={current_lr:.2e} train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f} | best@{best['epoch']}={best['val_acc']:.4f}")
        # step scheduler
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(va_loss)
            else:
                scheduler.step()
        # early stopping
        if args.early_stop_patience > 0 and (epoch - best_es_epoch) >= args.early_stop_patience:
            print(f"[EARLY-STOP] patience {args.early_stop_patience} reached (best epoch {best_es_epoch}).")
            break

    # post-run verification
    actual_epochs = 0
    if log_path.exists():
        with log_path.open('r', encoding='utf-8') as f:
            for _ in f:
                actual_epochs += 1
    if actual_epochs != args.epochs:
        meta2 = {'intended_epochs': int(args.epochs), 'actual_logged_epochs': int(actual_epochs)}
        save_json(meta2, str(run_dir / 'run_meta.json'))
        print(f"[WARN] intended epochs={args.epochs} but found only {actual_epochs} logged. Wrote run_meta.json to {run_dir}")

    generate_run_report(str(run_dir))


if __name__ == '__main__':
    RUN_ARGS = [
        '--npz', '/home/user/projects/train/train_data/slipce/windows_dense_npz.npz',
        '--use_norm',
        '--epochs', '40',
        '--batch_size', '128',
        '--lr', '5e-5',
        '--balance_by_class',
        '--hard_negative_factor', '1.0',  # 不加 --amplify_hard_negative 即代表關閉
        '--temporal_jitter_frames', '2',
        '--tcn_dropout', '0.2', '--fc_dropout', '0.4',
        '--label_smoothing', '0.05',
        '--lr_scheduler', 'plateau', '--plateau_patience', '5', '--plateau_factor', '0.5', '--min_lr', '1e-6',
        '--early_stop_patience', '12',
        '--tag', 'demo'
    ]
    argv = sys.argv[1:] if len(sys.argv) > 1 else RUN_ARGS
    main(argv)
