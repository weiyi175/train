#!/usr/bin/env python3
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
import torch.nn.functional as F

from dataset import LongWindowNPZDataset, collate_batch
from model import BiLSTMGlobalPooling

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/home/user/projects/train/train_data/slipce/windows_npz.npz')
    ap.add_argument('--out', default='/home/user/projects/train/model/long_window/BiLSTM_GlobalPooling/result_kfold')
    ap.add_argument('--k', type=int, default=5, help='number of folds')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--lr', type=float, default=1e-3)
    # only support CUDA device
    ap.add_argument('--device', choices=['cuda'], default='cuda')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=1)
    ap.add_argument('--pool', choices=['avg','max','attn'], default='avg')
    ap.add_argument('--use_bn', action='store_true')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def make_report(out_dir: Path, history, best_epoch_idx, args, device, ds, opt):
    # simplified report similar to train_bilstm
    try:
        param_count = sum(p.numel() for p in torch.load(out_dir / 'best.pt')['model_state'].values())
    except Exception:
        param_count = sum(p.numel() for p in ds.__class__.__name__ and [])
    try:
        N, T, F = ds.features.shape
    except Exception:
        N = len(ds); T = 0; F = 0
    best_entry = history[best_epoch_idx] if history else {}
    final_entry = history[-1] if history else {}
    lines = []
    lines.append('# K-Fold 訓練報告')
    lines.append(f'- fold_out: {out_dir}')
    lines.append(f'- 模型: BiLSTM_GlobalPooling | 裝置: {device} | 參數量: {param_count}')
    lines.append(f'- 資料: N={N} T={T} F={F} | 批次: {args.batch} | epoch: {args.epochs}')
    lines.append('')
    lines.append('## 核心指標')
    lines.append(f'- 最佳 (index): {best_epoch_idx} -> {best_entry}')
    lines.append(f'- 最終: {final_entry}')
    with open(out_dir / 'report.md','w') as f:
        f.write('\n'.join(lines))


def train_one_fold(train_idx, val_idx, fold_out: Path, args, device, ds):
    from torch.utils.data import Subset, DataLoader
    train_ds = Subset(ds, train_idx); val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    sample = ds[0]['features']; input_dim = sample.shape[1]
    model = BiLSTMGlobalPooling(input_dim, hidden_size=args.hidden, num_layers=args.num_layers, pooling=args.pool, use_bn=args.use_bn, single_logit=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0; history = []
    oof_probs = np.zeros((len(val_idx),), dtype=np.float32)
    accum_steps = max(1, int(getattr(args, 'accumulation_steps', 1)))
    for epoch in range(1, args.epochs+1):
        model.train(); t0=time.time(); losses=[]
        opt.zero_grad()
        step_idx = 0
        for b in train_loader:
            x=b['features'].to(device); mask=b['mask'].to(device); lengths=b['lengths']
            y=b['label'].to(device).float()
            logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            # scale loss for gradient accumulation
            loss = loss / accum_steps
            loss.backward()
            losses.append(float((loss * accum_steps).detach().cpu()))
            step_idx += 1
            if step_idx % accum_steps == 0:
                opt.step()
                opt.zero_grad()
        # after epoch, if leftover grads, step once
        if step_idx % accum_steps != 0:
            opt.step(); opt.zero_grad()
        # eval on val
        import sklearn.metrics as _m
        ys=[]; prs=[]
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x=b['features'].to(device); mask=b['mask'].to(device); lengths=b['lengths']
                y=b['label']
                logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
                pr = torch.sigmoid(logits).cpu()
                ys.append(y); prs.append(pr)
        if prs:
            y = torch.cat(ys).numpy(); pr = torch.cat(prs).numpy()
            try:
                auc = float(_m.roc_auc_score(y, pr))
            except Exception:
                auc = float('nan')
        else:
            auc = float('nan')
        loss_mean = float(np.mean(losses)) if losses else 0.0
        history.append({'epoch':epoch,'loss':loss_mean,'auc':auc})
        # save best
        if not np.isnan(auc) and auc>best_auc:
            best_auc=auc; torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'history': history}, fold_out / 'best.pt')

    # compute oof probs for this val set using best model
    # load best model
    best = torch.load(fold_out / 'best.pt', map_location=device)
    model.load_state_dict(best['model_state'])
    model.eval()
    all_prs=[]; all_ys=[]
    with torch.no_grad():
        for b in val_loader:
            x=b['features'].to(device); mask=b['mask'].to(device); lengths=b['lengths']
            y=b['label']
            logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
            pr = torch.sigmoid(logits).cpu().numpy()
            all_prs.append(pr); all_ys.append(y.numpy())
    if all_prs:
        all_pr = np.concatenate(all_prs)
        all_y = np.concatenate(all_ys)
    else:
        all_pr = np.array([]); all_y = np.array([])

    # save history and report
    with open(fold_out / 'history.json','w') as f:
        json.dump(history, f, indent=2)
    make_report(fold_out, history, 0, args, device, ds, opt)
    return best_auc, val_idx, all_y, all_pr


def main():
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    # enforce CUDA-only execution
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available on this machine')
    device = torch.device('cuda')
    print('device', device)

    ds = LongWindowNPZDataset(args.npz)
    N = len(ds)
    # collect labels
    labels = []
    for i in range(N): labels.append(int(ds[i]['label']))
    labels = np.array(labels)

    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    out_base = Path(args.out); out_base.mkdir(parents=True, exist_ok=True)

    oof_preds = np.zeros((N,), dtype=np.float32)
    oof_trues = np.zeros((N,), dtype=np.int64)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), labels), start=1):
        print('Fold', fold, 'train', len(train_idx), 'val', len(val_idx))
        fold_out = out_base / f'fold_{fold:02d}'; fold_out.mkdir(parents=True, exist_ok=True)
        best_auc, v_idx, ys, prs = train_one_fold(train_idx, val_idx, fold_out, args, device, ds)
        fold_metrics.append(best_auc)
        # store oof
        if prs.size>0:
            oof_preds[val_idx] = prs
            oof_trues[val_idx] = ys

    # aggregate OOF
    import sklearn.metrics as _m
    mask = (oof_preds!=0) | (oof_trues!=0)
    try:
        oof_auc = float(_m.roc_auc_score(oof_trues[mask], oof_preds[mask]))
    except Exception:
        oof_auc = float('nan')
    print('fold_metrics', fold_metrics)
    print('oof_auc', oof_auc)
    with open(out_base / 'kfold_summary.json','w') as f:
        json.dump({'folds': fold_metrics, 'oof_auc': oof_auc}, f, indent=2)


if __name__=='__main__':
    main()
