#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from dataset_statpool import StatPoolFeatureDataset, collate_batch
from model_statpool import StatPoolMLP
from utils_scaler import StandardScalerJSON


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature_dir', default='/home/user/projects/train/train_data/slipce/windows_npz.npz')
    ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'model' / 'long_window' / 'StatPool' / 'result'))
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--single_logit', action='store_true')
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda', help="Device to use ('cuda' or 'cpu'). Default 'cuda'.")
    ap.add_argument('--hidden_sizes', type=str, default='512,128', help='Comma-separated hidden sizes for MLP, e.g. "512,128"')
    ap.add_argument('--use_bn', action='store_true', help='Use BatchNorm1d between hidden layers')
    ap.add_argument('--fit_scaler', action='store_true')
    ap.add_argument('--balance_by_class', action='store_true')
    ap.add_argument('--focal_loss', action='store_true')
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--topk_loss', type=int, default=20, help='How many top high-loss file paths to record per epoch')
    return ap.parse_args()


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not existing:
        return base / '01'
    nums = sorted(int(p.name) for p in existing)
    return base / f"{nums[-1]+1:02d}"


def train():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_base = Path(args.out)
    out = next_run_dir(out_base)
    out.mkdir(parents=True, exist_ok=True)
    # choose device: try to use requested device, fallback to CPU if unavailable
    # choose device: try to use requested device, fallback to CPU if unavailable
    req = args.device
    if req == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA requested but not available: falling back to CPU')
        req = 'cpu'
    device = torch.device(req)
    print(f"[info] using device: {device}")
    if device.type == 'cuda':
        print('[info] cuda devices:', torch.cuda.device_count(), ' current:', torch.cuda.current_device())
        try:
            print('[info] device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
        except Exception:
            pass
        try:
            print('[info] mem_allocated (before):', torch.cuda.memory_allocated(device))
        except Exception:
            pass

    ds = StatPoolFeatureDataset(args.feature_dir)
    # optionally fit scaler
    scaler = None
    if args.fit_scaler:
        # collect up to 1000 samples
        Xs = []
        for i in range(min(len(ds), 1000)):
            with np.load(ds.files[i]) as d:
                Xs.append(d['features'])
        if len(Xs) == 0:
            print('[warn] no feature files found to fit scaler; skipping fit_scaler')
            scaler = None
        else:
            X = np.vstack([x for x in Xs])
            scaler = StandardScalerJSON()
            scaler.fit(X)
            scaler.to_json(out / 'scaler.json')
    else:
        # try load existing
        scp = out / 'scaler.json'
        if scp.exists():
            scaler = StandardScalerJSON.from_json(scp)

    # rebuild dataset with scaler
    ds = StatPoolFeatureDataset(args.feature_dir, scaler=scaler)
    synthetic_mode = False
    # if no files found, create a tiny synthetic dataset for smoke testing
    if len(ds) == 0:
        print('[warn] no feature files found in', args.feature_dir, '; creating synthetic dataset for smoke run')
        synthetic_mode = True
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, n=16, pooled_dim=50):
                self.n = n
                self.pooled_dim = pooled_dim
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                pooled = torch.randn(self.pooled_dim)
                label = torch.tensor(1 if idx % 2 == 0 else 0, dtype=torch.long)
                path = f'synthetic://sample/{idx:04d}.npz'
                return {'pooled': pooled, 'label': label, 'path': path}
        synth = SyntheticDataset(n=32, pooled_dim=50)
        # split
        n = len(synth); idx = np.arange(n); np.random.shuffle(idx)
        n_val = max(1, int(0.1 * n))
        val_idx = idx[:n_val]; train_idx = idx[n_val:]
        from torch.utils.data import Subset
        train_ds = Subset(synth, train_idx); val_ds = Subset(synth, val_idx)
    else:
    # split simple
        n = len(ds); idx = np.arange(n); np.random.shuffle(idx)
        n_val = max(1, int(0.1 * n))
        val_idx = idx[:n_val]; train_idx = idx[n_val:]
        from torch.utils.data import Subset
        train_ds = Subset(ds, train_idx); val_ds = Subset(ds, val_idx)

    # optional weighted sampler for class balancing
    from torch.utils.data import WeightedRandomSampler
    pin_memory = True if device.type == 'cuda' else False
    if args.balance_by_class:
        # compute per-sample weights from train_ds
        ys = [int(train_ds[i]['label']) for i in range(len(train_ds))]
        ys = np.array(ys)
        cnt = np.bincount(ys, minlength=2).astype(np.float32)
        cw = 1.0 / np.maximum(cnt, 1.0)
        sw = cw[ys]
        sampler = WeightedRandomSampler(weights=torch.tensor(sw, dtype=torch.double), num_samples=len(sw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=False, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=pin_memory)

    # infer input dim (use synthetic if created)
    if synthetic_mode:
        sample = synth[0]['pooled']
    else:
        sample = ds[0]['pooled']
    input_dim = sample.numel()
    # parse hidden sizes
    hs = [int(x) for x in args.hidden_sizes.split(',') if x.strip()] if args.hidden_sizes else [512, 128]
    model = StatPoolMLP(input_dim, hidden_sizes=hs, dropout=0.3, num_classes=2, single_logit=args.single_logit, use_bn=args.use_bn).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0; best_epoch = -1
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        # sync and start timer
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        train_losses = []
        per_sample_losses = []
        for b in train_loader:
            # non_blocking transfer if using pinned memory (pin_memory=True)
            x = b['pooled'].to(device, non_blocking=True)
            y = b['label'].to(device, non_blocking=True)
            logits = model(x)

            # compute per-sample losses (tensor shape: [batch]) then mean for backward
            per_sample = None
            if args.single_logit:
                # binary classification
                yf = y.float()
                logit_s = logits.squeeze(1).float()
                if args.focal_loss:
                    prob = torch.sigmoid(logit_s)
                    p_t = prob * yf + (1 - prob) * (1 - yf)
                    ce = F.binary_cross_entropy_with_logits(logit_s, yf, reduction='none')
                    per_sample = ce * (1 - p_t).pow(args.focal_gamma)
                else:
                    per_sample = F.binary_cross_entropy_with_logits(logit_s, yf, reduction='none')
                probs = torch.sigmoid(logit_s)
                preds = (probs >= 0.5).long()
            else:
                # multi-class
                if args.focal_loss:
                    logp = torch.log_softmax(logits, dim=-1)
                    p = torch.exp(logp)
                    ce = F.nll_loss(logp, y, reduction='none')
                    at = p.gather(1, y.unsqueeze(1)).squeeze(1)
                    per_sample = ce * (1 - at).pow(args.focal_gamma)
                else:
                    per_sample = F.cross_entropy(logits, y, reduction='none')
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = probs >= 0.5

            # safe fallback: if per_sample is None, treat whole-batch scalar
            if per_sample is None:
                batch_loss = torch.tensor(0.0, device=device)
            else:
                batch_loss = per_sample.mean()

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            train_losses.append(float(batch_loss.detach().cpu()))

            # record per-sample losses mapped to file paths when available
            try:
                if per_sample is not None:
                    # move per-sample losses to CPU once and convert to python floats
                    pl_list = per_sample.detach().cpu().tolist()
                    if isinstance(pl_list, list):
                        paths = b.get('path', None)
                        if paths is not None:
                            for pth, lv in zip(paths, pl_list):
                                per_sample_losses.append((pth, float(lv)))
            except Exception:
                pass
        # eval
        model.eval(); import sklearn.metrics as _m
        ys=[]; ps=[]; prs=[]
        with torch.no_grad():
            for b in val_loader:
                x=b['pooled'].to(device); y=b['label'].to(device)
                logits = model(x)
                if args.single_logit:
                    pr = torch.sigmoid(logits.squeeze(1))
                    p = (pr>=0.5).long()
                else:
                    pr = torch.softmax(logits,dim=-1)[:,1]
                    p = (pr>=0.5).long()
                ys.append(y.cpu()); ps.append(p.cpu()); prs.append(pr.cpu())
        if prs:
            y = torch.cat(ys).numpy(); p = torch.cat(ps).numpy(); pr = torch.cat(prs).numpy()
            try:
                auc = float(_m.roc_auc_score(y, pr))
            except Exception:
                auc = float('nan')
            f1 = float(_m.f1_score(y, p))
        else:
            auc = float('nan'); f1 = float('nan')
        loss_mean = float(np.mean(train_losses)) if train_losses else 0.0
        history.append({'epoch':epoch,'loss':loss_mean,'f1':f1,'auc':auc})
        # write top-k high loss samples
        if per_sample_losses:
            per_sample_losses.sort(key=lambda x: x[1], reverse=True)
            topk = per_sample_losses[:args.topk_loss]
            with open(out / f'topk_high_loss_epoch_{epoch:02d}.txt','w') as f:
                for pth,lv in topk:
                    f.write(f"{pth}\t{lv:.6f}\n")
        # sync before measuring time if on cuda
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"Epoch {epoch} loss={loss_mean:.4f} f1={f1:.4f} auc={auc:.4f} time={elapsed:.1f}s")
        if device.type == 'cuda':
            try:
                print('[info] mem_allocated (after):', torch.cuda.memory_allocated(device), 'peak:', torch.cuda.max_memory_allocated(device))
            except Exception:
                pass
        # save
        # save CPU-copy of state_dict to avoid device-specific tensors in file
        sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({'model_state': sd_cpu, 'epoch': epoch, 'history': history}, out / f"ckpt_epoch_{epoch:02d}.pt")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc; best_epoch = epoch
            torch.save({'model_state': sd_cpu, 'epoch': epoch, 'history': history}, out / 'best.pt')
    # write history
    with open(out / 'history.json','w') as f:
        json.dump(history, f, indent=2)
    print('Done. best_epoch', best_epoch, 'best_auc', best_auc)

if __name__=='__main__':
    train()
