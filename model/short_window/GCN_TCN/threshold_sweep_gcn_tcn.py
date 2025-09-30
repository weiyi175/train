#!/usr/bin/env python3
"""Threshold sweep for GCN_TCN runs.

Reads each run directory under --base (default: ./result) that contains
results_extended.json (with stored probs not currently) so we re-forward model if probs file absent.
Outputs per-run CSV and a summary CSV aggregating best metrics under different optimization targets.

Usage:
  python threshold_sweep_gcn_tcn.py --base result --test-npz /path/to/test_npz \
      --targets f1 composite precision_aware --min 0.05 --max 0.95 --step 0.01

If a run already has cached_probs.npz (keys: probs,y_true) we reuse.
Otherwise we load last.ckpt and run inference (requires same test npz + flags used: --use_norm assumed from config.json).

We compute metrics for each threshold t:
  precision, recall, f1, auc(once), composite=0.5*recall+0.3*f1+0.2*auc, precision_aware=0.5*precision+0.3*f1+0.2*auc

Per run we pick, for each target in targets, the threshold giving max target (ties -> lower threshold) and report.
"""
from __future__ import annotations
import argparse, json, os, math, csv
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix

# import project modules
ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'Tool'
if str(TOOL) not in os.sys.path:
    os.sys.path.append(str(TOOL))
from dataset_npz import WindowsNPZDataset  # type: ignore
from models import GCN_TCN_Classifier  # type: ignore

METRIC_FORMULAS = {
    'composite': '0.5*recall + 0.3*f1 + 0.2*auc',
    'precision_aware': '0.5*precision + 0.3*f1 + 0.2*auc'
}

def load_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'config.json missing in {run_dir}')
    return json.loads(cfg_path.read_text(encoding='utf-8'))

def forward_probs(run_dir: Path, cfg: Dict[str, Any], test_npz: Path, force: bool=False):
    cache = run_dir / 'cached_probs.npz'
    if cache.exists() and not force:
        data = np.load(cache)
        return data['probs'], data['y_true']
    use_norm = cfg.get('use_norm', False)
    test_ds = WindowsNPZDataset(npz_path=str(test_npz), split='short', use_norm=use_norm, temporal_jitter_frames=0)
    def collate(batch):
        xs,masks,ys,ws=[],[],[],[]
        for b in batch:
            x=b['x']; m=b.get('mask')
            if isinstance(x,np.ndarray): x=torch.from_numpy(x)
            if isinstance(m,np.ndarray): m=torch.from_numpy(m)
            if m is None: m_time=torch.ones(x.shape[0],dtype=torch.bool)
            else:
                mb=m.bool(); m_time=mb.any(dim=-1) if mb.ndim==2 else mb
            xs.append(torch.nan_to_num(x,nan=0.0)); masks.append(m_time); ys.append(int(b['y']))
        return {'x':torch.stack(xs,0),'mask':torch.stack(masks,0),'y':torch.tensor(ys)}
    loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate)
    in_dim = test_ds[0]['x'].shape[1]
    params = cfg['params'] if 'params' in cfg else {}
    model = GCN_TCN_Classifier(
        in_dim=in_dim, n_classes=2, gcn_hidden=params.get('gcn_hidden', 64),
        tcn_channels=tuple(int(x) for x in str(params.get('tcn_channels','128,128')).split(',')),
        tcn_kernel=params.get('tcn_kernel',3), tcn_dropout=params.get('tcn_dropout',0.1),
        tcn_dil_growth=params.get('tcn_dil_growth',2), fc_hidden=params.get('fc_hidden',128),
        fc_dropout=params.get('fc_dropout',0.2)
    )
    ckpt = run_dir / 'last.ckpt'
    if not ckpt.exists():
        raise FileNotFoundError(f'last.ckpt missing in {run_dir}')
    state = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()
    probs_list=[]; y_list=[]
    with torch.no_grad():
        for batch in loader:
            x=batch['x']
            m=batch['mask']
            logits,_ = model(x, mask=m)
            p = torch.softmax(logits, dim=-1)[:,1].cpu().numpy()
            probs_list.append(p)
            y_list.append(batch['y'].cpu().numpy())
    probs = np.concatenate(probs_list)
    y_true = np.concatenate(y_list)
    np.savez_compressed(cache, probs=probs, y_true=y_true)
    return probs, y_true

def sweep_thresholds(probs: np.ndarray, y_true: np.ndarray, thresholds: List[float]) -> List[Dict[str, Any]]:
    # precompute AUC once
    try:
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true))>1 else 0.0
    except Exception:
        auc = 0.0
    results=[]
    for t in thresholds:
        preds = (probs >= t).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        except ValueError:
            if preds.sum()==0:
                tp=fp=fn=0; tn=int(len(preds))
            else:
                tp=int(preds.sum()); fp=fn=0; tn=0
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = f1_score(y_true, preds, zero_division=0)
        composite = 0.5*recall + 0.3*f1 + 0.2*auc
        precision_aware = 0.5*precision + 0.3*f1 + 0.2*auc
        results.append({
            'threshold': t,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'composite': composite,
            'precision_aware': precision_aware,
            'TP': tp,'FP': fp,'FN': fn,'TN': tn
        })
    return results

def pick_best(results: List[Dict[str, Any]], target: str):
    best = None
    for r in results:
        if math.isnan(r[target]):
            continue
        if best is None or r[target] > best[target] or (r[target]==best[target] and r['threshold']<best['threshold']):
            best = r
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=str, default=str(Path(__file__).parent / 'result'))
    ap.add_argument('--test-npz', type=str, required=True, help='Independent test set npz path (dense).')
    ap.add_argument('--min', type=float, default=0.05)
    ap.add_argument('--max', type=float, default=0.95)
    ap.add_argument('--step', type=float, default=0.01)
    ap.add_argument('--targets', type=str, default='f1,composite,precision_aware')
    ap.add_argument('--force-forward', action='store_true', help='Ignore cached_probs and forward again.')
    args = ap.parse_args()

    thresholds = [round(x,4) for x in np.arange(args.min, args.max+1e-9, args.step)]
    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    base = Path(args.base)
    test_npz = Path(args.test_npz)
    run_dirs = sorted([p for p in base.iterdir() if p.is_dir() and (p/ 'config.json').exists()])

    summary_rows=[]
    for rd in run_dirs:
        try:
            cfg = load_config(rd)
        except Exception as e:
            print(f'[SKIP] {rd.name}: load_config error {e}')
            continue
        try:
            probs,y_true = forward_probs(rd, cfg, test_npz, force=args.force_forward)
        except Exception as e:
            print(f'[SKIP] {rd.name}: forward_probs error {e}')
            continue
        results = sweep_thresholds(probs, y_true, thresholds)
        out_csv = rd / 'threshold_metrics.csv'
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader(); writer.writerows(results)
        row_base = {'run': rd.name, 'seed': cfg.get('seed')}
        for tgt in targets:
            best = pick_best(results, tgt)
            if best:
                for k in ['threshold','auc','precision','recall','f1','composite','precision_aware']:
                    row_base[f'{tgt}_{k}'] = best[k]
        summary_rows.append(row_base)
        print(f'[DONE] {rd.name} thresholds swept; best metrics extracted.')

    # write summary
    if summary_rows:
        summary_csv = base / 'threshold_sweep_summary.csv'
        # collect all dynamic headers
        headers = sorted({h for r in summary_rows for h in r.keys()})
        with summary_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader(); writer.writerows(summary_rows)
        print(f'[SUMMARY] {summary_csv}')
    else:
        print('[WARN] No runs processed.')

if __name__ == '__main__':
    main()
