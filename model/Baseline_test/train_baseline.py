#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json, argparse, random
import numpy as np

# 路徑：加入 Tool 目錄以匯入 dataset_npz
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOL_DIR = os.path.join(ROOT, 'Tool')
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

from dataset_npz import WindowsNPZDataset, build_dataloader
from utils import ensure_incremental_run_dir, save_json, AvgMeter, count_params, write_lines


def _safe_float(x, default: float = float('nan')) -> float:
    try:
        return float(x)
    except Exception:
        return default


def generate_run_report(run_dir: str) -> None:
    """Read config.json and train_log.jsonl in a run_dir, analyze metrics, detect overfitting,
    and write report.json and report.md.

    Overfitting indicators (any 2+ imply overfitting=True):
      - best_epoch <= 0.6 * epochs
      - last_val_loss - best_val_loss >= max(0.1, 0.1 * best_val_loss)
      - gen_gap_last = (train_acc_last - val_acc_last) >= 0.15
      - best_val_acc - last_val_acc >= 0.02
    """
    cfg_path = os.path.join(run_dir, 'config.json')
    log_path = os.path.join(run_dir, 'train_log.jsonl')
    if not (os.path.exists(cfg_path) and os.path.exists(log_path)):
        return
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    rows = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return
    epochs = int(cfg.get('epochs', len(rows))) if isinstance(cfg.get('epochs', None), int) else len(rows)
    # Extract sequences
    ep = [int(r.get('epoch', i+1)) for i, r in enumerate(rows)]
    tr_loss = [ _safe_float(r.get('train_loss')) for r in rows ]
    tr_acc  = [ _safe_float(r.get('train_acc'))  for r in rows ]
    va_loss = [ _safe_float(r.get('val_loss'))   for r in rows ]
    va_acc  = [ _safe_float(r.get('val_acc'))    for r in rows ]
    # Best by val_acc
    import numpy as _np
    va_acc_np = _np.array(va_acc, dtype=float)
    best_idx = int(_np.nanargmax(va_acc_np))
    best_epoch = ep[best_idx]
    best = {
        'epoch': best_epoch,
        'train_loss': tr_loss[best_idx],
        'train_acc': tr_acc[best_idx],
        'val_loss': va_loss[best_idx],
        'val_acc': va_acc[best_idx],
    }
    last = {
        'epoch': ep[-1],
        'train_loss': tr_loss[-1],
        'train_acc': tr_acc[-1],
        'val_loss': va_loss[-1],
        'val_acc': va_acc[-1],
    }
    # Trends (simple finite differences over last K)
    K = max(2, min(10, len(ep)))
    def _slope(arr):
        if len(arr) < 2:
            return float('nan')
        return (arr[-1] - arr[-K]) / max(1, (ep[-1] - ep[-K]))
    slope = {
        'train_loss_slope_lastK': _slope(tr_loss),
        'train_acc_slope_lastK': _slope(tr_acc),
        'val_loss_slope_lastK': _slope(va_loss),
        'val_acc_slope_lastK': _slope(va_acc),
    }
    # Gaps
    gen_gap_best = (best['train_acc'] - best['val_acc']) if _np.isfinite(best['train_acc']) and _np.isfinite(best['val_acc']) else float('nan')
    gen_gap_last = (last['train_acc'] - last['val_acc']) if _np.isfinite(last['train_acc']) and _np.isfinite(last['val_acc']) else float('nan')
    # Overfitting signals
    cond_early_best = (best_epoch <= 0.6 * ep[-1])
    cond_loss_rebound = (last['val_loss'] - best['val_loss']) >= max(0.1, 0.1 * best['val_loss']) if _np.isfinite(last['val_loss']) and _np.isfinite(best['val_loss']) else False
    cond_gap_large = gen_gap_last >= 0.15 if _np.isfinite(gen_gap_last) else False
    cond_acc_drop = (best['val_acc'] - last['val_acc']) >= 0.02 if _np.isfinite(best['val_acc']) and _np.isfinite(last['val_acc']) else False
    signals = {
        'best_epoch_ratio': best_epoch / float(ep[-1]),
        'cond_early_best': bool(cond_early_best),
        'cond_loss_rebound': bool(cond_loss_rebound),
        'cond_gap_large': bool(cond_gap_large),
        'cond_acc_drop': bool(cond_acc_drop),
    }
    overfitting_score = int(cond_early_best) + int(cond_loss_rebound) + int(cond_gap_large) + int(cond_acc_drop)
    overfitting = overfitting_score >= 2
    # Summaries
    summary = {
        'config': cfg,
        'epochs_logged': len(ep),
        'best': best,
        'last': last,
        'generalization_gap': {'at_best': gen_gap_best, 'at_last': gen_gap_last},
        'trends': slope,
        'overfitting': {
            'is_overfitting': overfitting,
            'score': overfitting_score,
            'signals': signals,
        },
    }
    # Save JSON
    with open(os.path.join(run_dir, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    # Save Markdown
    def _fmt(v, nd=4):
        try:
            return f"{float(v):.{nd}f}"
        except Exception:
            return str(v)
    md = []
    md.append(f"# 訓練報告\n")
    md.append(f"- 模型: {cfg.get('model')}  | 分割: {cfg.get('split')}  | 裝置: {cfg.get('device')}  | 參數量: {cfg.get('params')}\n")
    md.append(f"- 資料: N={cfg.get('N')} T={cfg.get('T')} F={cfg.get('F')}  | 批次: {cfg.get('batch_size')}  | epoch: {cfg.get('epochs')}\n")
    md.append(f"\n## 核心指標\n")
    md.append(f"- 最佳 (epoch {best['epoch']}): train_loss={_fmt(best['train_loss'])}, train_acc={_fmt(best['train_acc'])}, val_loss={_fmt(best['val_loss'])}, val_acc={_fmt(best['val_acc'])}\n")
    md.append(f"- 最終 (epoch {last['epoch']}): train_loss={_fmt(last['train_loss'])}, train_acc={_fmt(last['train_acc'])}, val_loss={_fmt(last['val_loss'])}, val_acc={_fmt(last['val_acc'])}\n")
    md.append(f"- 一般化落差: at_best={_fmt(gen_gap_best)}, at_last={_fmt(gen_gap_last)}\n")
    md.append(f"\n## 趨勢 (最後 {K} 個 epoch 粗略斜率)\n")
    md.append(f"- train_loss_slope: {_fmt(slope['train_loss_slope_lastK'])}\n")
    md.append(f"- train_acc_slope: {_fmt(slope['train_acc_slope_lastK'])}\n")
    md.append(f"- val_loss_slope: {_fmt(slope['val_loss_slope_lastK'])}\n")
    md.append(f"- val_acc_slope: {_fmt(slope['val_acc_slope_lastK'])}\n")
    md.append(f"\n## 過擬合分析\n")
    md.append(f"- 判定: {'是' if overfitting else '否'} (score={overfitting_score})\n")
    md.append(f"- 訊號: early_best={signals['cond_early_best']}, loss_rebound={signals['cond_loss_rebound']}, gap_large={signals['cond_gap_large']}, acc_drop={signals['cond_acc_drop']}\n")
    md.append(f"- 附註: best_epoch_ratio={_fmt(signals['best_epoch_ratio'], 2)}\n")
    md.append(f"\n## 設定摘要\n")
    for k in ['lr','weight_decay','seed','use_norm','concat_raw_norm','balance_by_class','amplify_hard_negative','hard_negative_factor','temporal_jitter_frames','val_ratio','num_workers']:
        if k in cfg:
            md.append(f"- {k}: {cfg[k]}\n")
    with open(os.path.join(run_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(''.join(md))


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


def build_model(name: str, T: int, F: int, hidden: int, dropout: float):
    import torch.nn as nn
    if name == 'mlp_flat':
        from models import MLPFlat
        return MLPFlat(T, F, hidden=hidden, dropout=dropout)
    elif name == 'statpool_mlp':
        from models import StatPoolMLP
        return StatPoolMLP(F, hidden=hidden, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {name}")


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
        # replace NaN with 0 using mask if available
        x = x.clone()
        x[torch.isnan(x)] = 0.0
        logits = model(x, mask=mask)
        # CE with sample weights
        loss_vec = F.cross_entropy(logits, y, reduction='none')
        if cls_weighted_loss:
            loss = (loss_vec * w).sum() / (w.sum().clamp(min=1.0))
        else:
            loss = loss_vec.mean()
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
    ap = argparse.ArgumentParser(description='Baseline_train: MLP baselines on dense NPZ')
    ap.add_argument('--npz', default=get_default_npz_path())
    ap.add_argument('--split', choices=['short','long'], default='short')
    ap.add_argument('--use_norm', action='store_true', help='use normalized features as input')
    ap.add_argument('--model', choices=['mlp_flat','statpool_mlp'], default='mlp_flat')
    ap.add_argument('--concat_raw_norm', action='store_true', help='將 raw 與 norm 在特徵維度串接後輸入模型')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--temporal_jitter_frames', type=int, default=0)
    ap.add_argument('--balance_by_class', action='store_true', help='use class-balanced sampler')
    ap.add_argument('--amplify_hard_negative', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=1.0)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--dry_run', action='store_true', help='load and print shapes only (no torch import)')
    args = ap.parse_args()

    # Load dataset (numpy only here)
    ds = WindowsNPZDataset(args.npz, split=args.split, use_norm=args.use_norm, temporal_jitter_frames=args.temporal_jitter_frames)
    N, T, F = ds.x.shape
    print(f"Dataset loaded: split={args.split} N={N} T={T} F={F}; use_norm={args.use_norm}")
    if args.concat_raw_norm:
        d = np.load(args.npz, allow_pickle=True)
        key = 'short' if args.split=='short' else 'long'
        raw = d[f'{key}_raw']
        norm = d[f'{key}_norm']
        mask = d[f'{key}_mask']
        y = d[f'{key}_label']
        w = d[f'{key}_weight'] if f'{key}_weight' in d else np.ones((raw.shape[0],), dtype=np.float32)
        X = np.concatenate([raw, norm], axis=-1)
        M = np.concatenate([mask, mask], axis=-1)
        N, T, F = X.shape
        print(f"Concat enabled -> new F={F} (raw{raw.shape[-1]}+norm{norm.shape[-1]})")

    # Dry run: exit before importing torch/models
    if args.dry_run:
        return

    set_seed(args.seed)

    try:
        import torch
    except Exception as e:
        print('PyTorch 未安裝或不可用，請先安裝，例如： pip install torch --index-url https://download.pytorch.org/whl/cpu')
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split train/val
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    import importlib
    torch_data = importlib.import_module('torch.utils.data')
    if args.concat_raw_norm:
        # Custom lightweight dataset
        class _ConcatDS:
            def __init__(self, X, M, y, w, idx):
                self.X = X; self.M = M; self.y = y; self.w = w; self.idx = np.array(idx)
            def __len__(self): return len(self.idx)
            def __getitem__(self, i):
                import torch
                j = int(self.idx[i])
                return {
                    'x': torch.from_numpy(self.X[j].astype(np.float32)),
                    'mask': torch.from_numpy(self.M[j].astype(np.bool_)),
                    'y': torch.tensor(int(self.y[j]), dtype=torch.long),
                    'weight': torch.tensor(float(self.w[j]), dtype=torch.float32),
                }
        # Sampler weights similar to dataset_npz.build_sampler
        def _build_sampler(idx, y, w):
            import torch
            Nn = len(idx)
            if not args.balance_by_class and not args.amplify_hard_negative:
                return None
            ys = y[idx]
            ws = w[idx].astype(np.float32)
            cw = np.ones_like(ws, dtype=np.float32)
            if args.balance_by_class:
                pos = (ys==1).sum(); neg = (ys==0).sum()
                if pos>0 and neg>0:
                    w_pos = neg / (pos + neg)
                    w_neg = pos / (pos + neg)
                    cw = np.where(ys==1, w_pos, w_neg).astype(np.float32)
            if args.amplify_hard_negative:
                hn_mask = (ws < 1.0)
                cw = cw * np.where(hn_mask, float(args.hard_negative_factor), 1.0)
            final_w = cw * ws
            WeightedRandomSampler = getattr(torch_data, 'WeightedRandomSampler')
            return WeightedRandomSampler(importlib.import_module('torch').from_numpy(final_w), num_samples=Nn, replacement=True)

        ds_train = _ConcatDS(X, M, y, w, train_idx)
        ds_val = _ConcatDS(X, M, y, w, val_idx)
        sampler = _build_sampler(train_idx, y, w)
        dl_train = torch_data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                                         num_workers=args.num_workers, pin_memory=True, drop_last=False)
        dl_val = torch_data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=False)
    else:
        # Subset-like view via keep_idx override
        ds_train = WindowsNPZDataset(args.npz, split=args.split, use_norm=args.use_norm, temporal_jitter_frames=args.temporal_jitter_frames)
        ds_train.keep_idx = train_idx
        ds_val = WindowsNPZDataset(args.npz, split=args.split, use_norm=args.use_norm, temporal_jitter_frames=0)
        ds_val.keep_idx = val_idx
        from dataset_npz import build_dataloader
        dl_train = build_dataloader(ds_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                    balance_by_class=args.balance_by_class, amplify_hard_negative=args.amplify_hard_negative,
                                    hard_negative_factor=args.hard_negative_factor)
        dl_val = torch_data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                       pin_memory=True, drop_last=False)

    # Build model
    model = build_model(args.model, T, F, hidden=args.hidden, dropout=args.dropout)
    model.to(device)
    n_params = count_params(model)

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Prepare run dir
    result_base = os.path.join(ROOT, 'model', 'Baseline_test', 'result')
    run_dir = ensure_incremental_run_dir(result_base)
    print(f"Run dir: {run_dir}")
    config = vars(args).copy(); config.update({'N': int(N), 'T': int(T), 'F': int(F), 'params': int(n_params), 'device': str(device)})
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
            # save best
            if va_acc > best_acc:
                best_acc = va_acc
                best_path = os.path.join(run_dir, 'best.ckpt')
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': va_acc}, best_path)
        # always save last
        torch.save({'model': model.state_dict(), 'epoch': args.epochs, 'val_acc': va_acc}, os.path.join(run_dir, 'last.ckpt'))
    # 回填 epochs 與 batch_size 到 config 供報告使用
    try:
        cfg_update = json.load(open(os.path.join(run_dir, 'config.json'), 'r', encoding='utf-8'))
        cfg_update['epochs'] = int(args.epochs)
        cfg_update['batch_size'] = int(args.batch_size)
        save_json(cfg_update, os.path.join(run_dir, 'config.json'))
    except Exception:
        pass
    # 產生報告
    try:
        generate_run_report(run_dir)
    except Exception as e:
        print(f"[warn] 報告產生失敗: {e}")
    print(f"Saved run to: {run_dir}; best_acc={best_acc:.4f}; best_path={best_path}")


if __name__ == '__main__':
    main()
