#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

# add Tool to path for dataset
ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'Tool'
if str(TOOL) not in sys.path:
    sys.path.append(str(TOOL))
from dataset_npz import WindowsNPZDataset  # type: ignore

# local modules
from models import TCNWithAttentionClassifier
from utils import get_next_run_dir, save_json, count_params, generate_run_report


def build_train_loader(npz_path: str, use_norm: bool, batch_size: int, num_workers: int,
                       seed: int, balance_by_class: bool, amplify_hard_negative: bool, hard_negative_factor: float,
                       temporal_jitter_frames: int) -> Tuple[DataLoader, Dict[str, Any]]:
    ds = WindowsNPZDataset(npz_path=npz_path, split='short', use_norm=use_norm, temporal_jitter_frames=temporal_jitter_frames)
    N = len(ds)
    idx = list(range(N))
    ys = np.array([ds[i]['y'] for i in idx], dtype=np.int64)
    base_w = np.array([float(ds[i].get('weight', 1.0)) for i in idx], dtype=np.float32)
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
            if m is None:
                m_time = torch.ones(x.shape[0], dtype=torch.bool)
            else:
                m_bool = m.bool(); m_time = m_bool.any(dim=-1) if m_bool.ndim == 2 else m_bool
            xs.append(torch.nan_to_num(x, nan=0.0)); masks.append(m_time); ys.append(int(b['y'])); ws.append(float(b.get('weight',1.0)))
        return {'x': torch.stack(xs,0), 'mask': torch.stack(masks,0), 'y': torch.tensor(ys), 'weight': torch.tensor(ws)}

    train_loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    T, F = ds[0]['x'].shape
    return train_loader, {'N': N, 'T': T, 'F': F}


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
def evaluate(model, loader, device):
    model.eval()
    probs_list = []
    y_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device, non_blocking=True)
            m = batch['mask'].to(device, non_blocking=True)
            y = batch['y'].to(device, non_blocking=True)
            logits, _ = model(x, mask=m)
            p = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy().tolist()
            probs_list.extend(p)
            y_list.extend(y.detach().cpu().numpy().tolist())
    return np.array(probs_list), np.array(y_list)


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    # default updated to slipce_thresh040 dataset
    ap.add_argument('--npz', default=str(ROOT / 'train_data' / 'slipce_thresh040' / 'windows_dense_npz.npz'))
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_ratio', type=float, default=0.2)  # deprecated in clean mode
    ap.add_argument('--test_npz', required=True, help='Independent test npz path')
    ap.add_argument('--no_val', action='store_true', help='Use entire npz for training (train+val merged), only final test evaluation.')
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

    if not args.no_val:
        print('[WARN] Clean split 建議使用 --no_val；目前仍使用舊流程 (含 val)，若要最乾淨請加 --no_val。')
        # fallback minimal val split
        ds_full = WindowsNPZDataset(npz_path=args.npz, split='short', use_norm=args.use_norm, temporal_jitter_frames=args.temporal_jitter_frames)
        N=len(ds_full); g=torch.Generator().manual_seed(args.seed); idx=torch.randperm(N,generator=g).tolist(); n_val=max(1,int(N*args.val_ratio)); val_idx=idx[:n_val]; train_idx=idx[n_val:]
        def collate(batch):
            xs,masks,ys,ws=[],[],[],[]
            for b in batch:
                x=b['x']; m=b.get('mask')
                if isinstance(x,np.ndarray): x=torch.from_numpy(x)
                if isinstance(m,np.ndarray): m=torch.from_numpy(m)
                if m is None: m_time=torch.ones(x.shape[0],dtype=torch.bool)
                else:
                    mb=m.bool(); m_time=mb.any(dim=-1) if mb.ndim==2 else mb
                xs.append(torch.nan_to_num(x,nan=0.0)); masks.append(m_time); ys.append(int(b['y'])); ws.append(float(b.get('weight',1.0)))
            return {'x':torch.stack(xs,0),'mask':torch.stack(masks,0),'y':torch.tensor(ys),'weight':torch.tensor(ws)}
        train_ds=torch.utils.data.Subset(ds_full,train_idx); val_ds=torch.utils.data.Subset(ds_full,val_idx)
        tr_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
        va_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
        T,F=ds_full[0]['x'].shape; meta={'N':N,'T':T,'F':F}
    else:
        tr_loader, meta = build_train_loader(
            npz_path=args.npz,
            use_norm=args.use_norm,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            balance_by_class=args.balance_by_class,
            amplify_hard_negative=args.amplify_hard_negative,
            hard_negative_factor=args.hard_negative_factor,
            temporal_jitter_frames=args.temporal_jitter_frames,
        )
        va_loader=None

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

    best = {'train_acc': -1.0, 'epoch': -1}
    log_path = run_dir / 'train_log.jsonl'
    history = []
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, device, criterion, optim)
        rec = {'epoch': epoch, 'train_loss': float(tr_loss), 'train_acc': float(tr_acc)}
        history.append(rec)
        with log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec)+"\n")
        if tr_acc > best['train_acc']:
            best = {'train_acc': tr_acc, 'epoch': epoch}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'train_acc': tr_acc}, run_dir / 'best.ckpt')
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'train_acc': tr_acc}, run_dir / 'last.ckpt')
        print(f"[E{epoch:03d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | best@{best['epoch']}={best['train_acc']:.4f}")

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

    if not args.no_val:
        generate_run_report(str(run_dir))
    else:
        # create minimal clean report (no validation metrics)
        rpt_path = run_dir / 'report.md'
        lines = [
            '# 訓練報告 (clean no-val 模式)',
            f"- 模型: tcn_attention | 裝置: {device} | 參數量: {num_params}",
            f"- 訓練資料: N={meta.get('N')} T={meta.get('T')} F={meta.get('F')} | epochs={args.epochs} | batch={args.batch_size}",
            '## Epoch 訓練紀錄 (train_loss / train_acc)',
        ]
        for r in history:
            lines.append(f"- epoch {r['epoch']}: loss={r['train_loss']:.4f}, acc={r['train_acc']:.4f}")
        with open(rpt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    # Independent test evaluation
    if args.test_npz and os.path.exists(args.test_npz):
        test_ds = WindowsNPZDataset(npz_path=args.test_npz, split='short', use_norm=args.use_norm, temporal_jitter_frames=0)
        def collate(batch):
            xs,masks,ys,ws=[],[],[],[]
            for b in batch:
                x=b['x']; m=b.get('mask')
                if isinstance(x,np.ndarray): x=torch.from_numpy(x)
                if isinstance(m,np.ndarray): m=torch.from_numpy(m)
                if m is None: m_time=torch.ones(x.shape[0],dtype=torch.bool)
                else:
                    mb=m.bool(); m_time=mb.any(dim=-1) if mb.ndim==2 else mb
                xs.append(torch.nan_to_num(x,nan=0.0)); masks.append(m_time); ys.append(int(b['y'])); ws.append(float(b.get('weight',1.0)))
            return {'x':torch.stack(xs,0),'mask':torch.stack(masks,0),'y':torch.tensor(ys),'weight':torch.tensor(ws)}
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)
        probs, y_true = evaluate(model, test_loader, device)
        preds = (probs >= 0.5).astype(int)
        try:
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, probs))
            else:
                auc = 0.0
        except Exception:
            auc = 0.0
        f1 = float(f1_score(y_true, preds, zero_division=0))
        recall = float(recall_score(y_true, preds, zero_division=0))
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        except ValueError:
            if preds.sum() == 0:
                tp = fp = fn = 0; tn = int(len(preds))
            else:
                tp = int(preds.sum()); fp = fn = 0; tn = 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        composite = 0.5 * recall + 0.3 * f1 + 0.2 * auc
        precision_aware = 0.5 * precision + 0.3 * f1 + 0.2 * auc
        results = {
            'test': {
                'auc': auc, 'f1': f1, 'recall': recall, 'precision': precision,
                'composite': composite, 'precision_aware': precision_aware,
                'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
            },
            'top_epochs': 'N/A (clean split, no validation ranking)',
            'top_epochs_precision_aware': 'N/A (clean split, no validation ranking)',
            'params': cfg
        }
        with open(run_dir / 'results_extended.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        rpt_path = run_dir / 'report.md'
        try:
            with open(rpt_path, 'a', encoding='utf-8') as f:
                f.write('\n\n## Test metrics (independent)\n')
                f.write(f"- AUC: {auc:.4f}\n")
                f.write(f"- F1: {f1:.4f}\n")
                f.write(f"- Recall: {recall:.4f}\n")
                f.write(f"- Precision: {precision:.4f}\n")
                f.write(f"- Composite Score: {composite:.4f} (0.5*Recall + 0.3*F1 + 0.2*AUC)\n")
                f.write(f"- Precision-aware Score: {precision_aware:.4f} (0.5*Precision + 0.3*F1 + 0.2*AUC)\n")
                f.write('## Confusion matrix (TP/FP/FN/TN)\n')
                f.write(f"- TP: {tp}\n- FP: {fp}\n- FN: {fn}\n- TN: {tn}\n\n")
                f.write('## Top 4 epochs by Composite\n- N/A (no validation set used)\n')
                f.write('\n## Top 4 epochs by Precision-aware\n- N/A (no validation set used)\n')
        except Exception as e:
            print('[WARN] failed to append test metrics to report.md ->', e)


if __name__ == '__main__':
    # Quick-run arguments updated to slipce_thresh040 dataset (clean split)
    RUN_ARGS = [
        '--npz', '/home/user/projects/train/train_data/slipce_thresh040/windows_dense_npz.npz',
        '--test_npz', '/home/user/projects/train/test_data/slipce_thresh040/windows_dense_npz.npz',
        '--no_val', '--use_norm', '--epochs', '20', '--batch_size', '256', '--lr', '1e-4',
        '--balance_by_class', '--amplify_hard_negative', '--hard_negative_factor', '1.5'
    ]
    argv = sys.argv[1:] if len(sys.argv) > 1 else RUN_ARGS
    main(argv)
