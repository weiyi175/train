#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

from mix_dataset import OfflineFusionDataset
from models_mixed import FUSION_BUILDERS, WeightedAvgFusion


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short-probs', required=True)
    ap.add_argument('--long-probs', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--fusion', choices=['avg','weighted','mlp','stack_logistic'], default='weighted')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sweep-thresholds', action='store_true')
    ap.add_argument('--out', default='result_mix')
    ap.add_argument('--no-train', action='store_true', help='For fusion=avg: skip training entirely')
    ap.add_argument('--save-probs', action='store_true', help='Export fused probs to fused_probs.npy')
    return ap.parse_args()


def build_model(name: str):
    if name == 'avg':
        return None  # handled separately
    if name in FUSION_BUILDERS:
        return FUSION_BUILDERS[name]()
    raise ValueError(f'Unknown fusion={name}')


def train_epoch(model, loader, device, optim, criterion):
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        # ensure float32 tensor shape (B,2)
        p_short = batch['p_short'].to(device).float()
        p_long = batch['p_long'].to(device).float()
        p = torch.stack([p_short, p_long], dim=1)
        y = batch['y'].to(device)
        logits = model(p)
        loss = criterion(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        bsz = y.size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz
    return total_loss / max(1, total_n)

@torch.no_grad()
def infer_probs(model_or_none, dataset, device, batch_size=512, fusion_name='weighted'):
    if fusion_name=='avg':
        # simple average
        p = 0.5*(dataset.short + dataset.long)
        return p.astype('float32')
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_p=[]
    for batch in dl:
        ps = torch.stack([batch['p_short'], batch['p_long']], dim=1).to(device).float()
        logits = model_or_none(ps)  # (B,2)
        prob = torch.softmax(logits, dim=-1)[:,1].detach().cpu().numpy()
        all_p.append(prob)
    return np.concatenate(all_p,0)

@torch.no_grad()
def evaluate_probs(probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
    preds = (probs >= threshold).astype(int)
    if len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, probs))
        except Exception:
            auc = 0.0
    else:
        auc = 0.0
    from sklearn.metrics import f1_score, recall_score, confusion_matrix
    f1 = float(f1_score(y_true, preds, zero_division=0))
    recall = float(recall_score(y_true, preds, zero_division=0))
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    except ValueError:
        if preds.sum()==0:
            tp=fp=fn=0; tn=int(len(preds))
        else:
            tp=int(preds.sum()); fp=fn=0; tn=0
    precision = tp / (tp + fp) if (tp+fp)>0 else 0.0
    composite = 0.5*recall + 0.3*f1 + 0.2*auc
    precision_aware = 0.5*precision + 0.3*f1 + 0.2*auc
    return dict(auc=auc,f1=f1,recall=recall,precision=precision,composite=composite,precision_aware=precision_aware,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))


def threshold_sweep(probs, y, out_dir: Path):
    rows=[]; best=None; best_pa=None
    for t in np.linspace(0.05,0.95,19):
        m = evaluate_probs(probs,y,threshold=float(t))
        m['threshold']=float(t)
        rows.append(m)
        if best is None or m['composite']>best['composite']:
            best=m
        if best_pa is None or m['precision_aware']>best_pa['precision_aware']:
            best_pa=m
    import csv
    with open(out_dir/'threshold_metrics.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    json.dump({'best_composite':best,'best_precision_aware':best_pa}, open(out_dir/'threshold_best.json','w'), ensure_ascii=False, indent=2)
    return best, best_pa


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ds = OfflineFusionDataset(args.short_probs, args.long_probs, args.labels)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # split train/val
    val_n = int(len(ds)*args.val_ratio)
    train_n = len(ds)-val_n
    tr_ds, va_ds = random_split(ds, [train_n,val_n], generator=torch.Generator().manual_seed(args.seed)) if val_n>0 else (ds,None)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.fusion)
    if model is not None:
        model.to(device)
    if args.fusion=='avg':
        print('[INFO] avg fusion does not train; directly evaluates.')
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        if not args.no_train:
            for epoch in range(1, args.epochs+1):
                loss = train_epoch(model, DataLoader(tr_ds, batch_size=args.batch, shuffle=True), device, optim, criterion)
                print(f'[E{epoch:03d}] loss={loss:.4f}')
        torch.save({'model': model.state_dict()}, out_dir/'fusion.ckpt')

    # inference on full dataset (post-train)
    probs = infer_probs(model, ds, device, fusion_name=args.fusion)
    y = ds.labels
    base_metrics = evaluate_probs(probs, y, 0.5)
    print('[BASE @0.5]', base_metrics)
    json.dump({'base@0.5': base_metrics, 'cfg': vars(args)}, open(out_dir/'fusion_results.json','w'), ensure_ascii=False, indent=2)

    if args.sweep_thresholds:
        best, best_pa = threshold_sweep(probs, y, out_dir)
        print('[SWEEP best_composite]', best)
        print('[SWEEP best_precision_aware]', best_pa)

    if args.save_probs:
        np.save(out_dir/'fused_probs.npy', probs.astype('float32'))

if __name__ == '__main__':
    main()
