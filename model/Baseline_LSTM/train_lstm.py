#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, argparse, random
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOL_DIR = os.path.join(ROOT, 'Tool')
BASELINE_TEST_DIR = os.path.join(ROOT, 'model', 'Baseline_test')
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)
if BASELINE_TEST_DIR not in sys.path:
    sys.path.insert(0, BASELINE_TEST_DIR)

from dataset_npz import WindowsNPZDataset, build_dataloader
from utils import ensure_incremental_run_dir, save_json, AvgMeter, count_params, write_lines
from train_baseline import generate_run_report  # 重用報告


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_default_npz_path() -> str:
    return os.path.join(ROOT, 'train_data', 'slipce', 'windows_dense_npz.npz')


def train_one_epoch(model, dl, optimizer, device, cls_weighted_loss: bool = True):
    import torch
    import torch.nn.functional as F
    model.train()
    meter_loss, meter_acc = AvgMeter(), AvgMeter()
    for batch in dl:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        w = batch['weight'].to(device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)
        x = x.clone()
        x[torch.isnan(x)] = 0.0
        logits = model(x, mask=mask)
        loss_vec = F.cross_entropy(logits, y, reduction='none')
        loss = (loss_vec * w).sum() / (w.sum().clamp(min=1.0)) if cls_weighted_loss else loss_vec.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == y).float().mean().item()
        meter_loss.update(float(loss.item()), x.size(0))
        meter_acc.update(acc, x.size(0))
    return meter_loss.avg, meter_acc.avg


def evaluate(model, dl, device):
    import torch
    import torch.nn.functional as F
    model.eval()
    meter_loss, meter_acc = AvgMeter(), AvgMeter()
    for batch in dl:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        w = batch['weight'].to(device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)
        x = x.clone()
        x[torch.isnan(x)] = 0.0
        logits = model(x, mask=mask)
        loss_vec = F.cross_entropy(logits, y, reduction='none')
        loss = (loss_vec * w).sum() / (w.sum().clamp(min=1.0))
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        meter_loss.update(float(loss.item()), x.size(0))
        meter_acc.update(acc, x.size(0))
    return meter_loss.avg, meter_acc.avg


def main():
    ap = argparse.ArgumentParser(description='Baseline LSTM on dense NPZ')
    ap.add_argument('--npz', default=get_default_npz_path())
    ap.add_argument('--split', choices=['short','long'], default='short')
    ap.add_argument('--use_norm', action='store_true')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=1)
    ap.add_argument('--bidirectional', action='store_true')
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--temporal_jitter_frames', type=int, default=0)
    ap.add_argument('--balance_by_class', action='store_true')
    ap.add_argument('--amplify_hard_negative', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=1.0)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    args = ap.parse_args()

    ds = WindowsNPZDataset(args.npz, split=args.split, use_norm=args.use_norm, temporal_jitter_frames=args.temporal_jitter_frames)
    N, T, F = ds.x.shape
    print(f"Dataset loaded: split={args.split} N={N} T={T} F={F}; use_norm={args.use_norm}")

    set_seed(args.seed)
    try:
        import torch
    except Exception:
        print('PyTorch 不可用，請先安裝')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # split
    idx = np.arange(N); np.random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_ratio))
    val_idx = idx[:n_val]; train_idx = idx[n_val:]

    ds_train = WindowsNPZDataset(args.npz, split=args.split, use_norm=args.use_norm, temporal_jitter_frames=args.temporal_jitter_frames)
    ds_train.keep_idx = train_idx
    ds_val = WindowsNPZDataset(args.npz, split=args.split, use_norm=args.use_norm, temporal_jitter_frames=0)
    ds_val.keep_idx = val_idx
    dl_train = build_dataloader(ds_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                balance_by_class=args.balance_by_class, amplify_hard_negative=args.amplify_hard_negative,
                                hard_negative_factor=args.hard_negative_factor)
    import importlib
    torch_data = importlib.import_module('torch.utils.data')
    dl_val = torch_data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=False)

    from lstm_models import LSTMClassifier
    model = LSTMClassifier(input_dim=F, hidden_size=args.hidden, num_layers=args.num_layers,
                           bidirectional=args.bidirectional, dropout=args.dropout).to(device)
    n_params = count_params(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # run dir
    result_base = os.path.join(ROOT, 'model', 'Baseline_LSTM', 'result')
    run_dir = ensure_incremental_run_dir(result_base)
    print(f"Run dir: {run_dir}")
    config = vars(args).copy(); config.update({'N': int(N), 'T': int(T), 'F': int(F), 'params': int(n_params), 'device': str(device), 'model':'lstm'})
    save_json(config, os.path.join(run_dir, 'config.json'))
    write_lines(os.path.join(run_dir, 'model_spec.txt'), [str(model), f"\nParams: {n_params}"])

    best_acc, best_path = -1.0, None
    log_path = os.path.join(run_dir, 'train_log.jsonl')
    with open(log_path, 'w', encoding='utf-8') as flog:
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = train_one_epoch(model, dl_train, optim, device)
            va_loss, va_acc = evaluate(model, dl_val, device)
            row = {'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc}
            print(json.dumps(row, ensure_ascii=False))
            flog.write(json.dumps(row, ensure_ascii=False)+"\n"); flog.flush()
            if va_acc > best_acc:
                best_acc = va_acc
                best_path = os.path.join(run_dir, 'best.ckpt')
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': va_acc}, best_path)
        torch.save({'model': model.state_dict(), 'epoch': args.epochs, 'val_acc': va_acc}, os.path.join(run_dir, 'last.ckpt'))

    # 填入 epochs/batch_size 供報告
    try:
        cfg_update = json.load(open(os.path.join(run_dir, 'config.json'), 'r', encoding='utf-8'))
        cfg_update['epochs'] = int(args.epochs)
        cfg_update['batch_size'] = int(args.batch_size)
        save_json(cfg_update, os.path.join(run_dir, 'config.json'))
    except Exception:
        pass
    # 產報告
    try:
        generate_run_report(run_dir)
    except Exception as e:
        print(f"[warn] 報告產生失敗: {e}")
    print(f"Saved run to: {run_dir}; best_acc={best_acc:.4f}; best_path={best_path}")


if __name__ == '__main__':
    main()
