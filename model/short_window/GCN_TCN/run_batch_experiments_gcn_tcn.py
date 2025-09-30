#!/usr/bin/env python3
"""Run curated batch-size experiments with tailored LR / focal / cosine schedule.

Configuration logic:
- Core set: 40,48,56 (lr=1e-4 cosine)
- Controls: 64 (lr=9e-5 cosine), 32 (lr=7e-5 cosine), 96 (lr=1e-4 cosine)
- Optional small noisy: 16 (lr=5e-5 cosine, focal_gamma=1.5)

Outputs each run into standard result/ auto indexing; collects metrics into experiments_batch_refined.csv
and (optionally) triggers threshold sweep summary after all runs.
"""
from __future__ import annotations
import subprocess, sys, json, csv
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[3]
THIS = Path(__file__).parent
TRAIN = THIS / 'train_gcn_tcn.py'
RESULT = THIS / 'result'

PLAN = [
    {'batch':40,'lr':1e-4,'focal':0.0},
    {'batch':48,'lr':1e-4,'focal':0.0},
    {'batch':56,'lr':1e-4,'focal':0.0},
    {'batch':64,'lr':9e-5,'focal':0.0},
    {'batch':32,'lr':7e-5,'focal':0.0},
    {'batch':96,'lr':1e-4,'focal':0.0},
    {'batch':16,'lr':5e-5,'focal':1.5},
]

def parse_results(run_dir: Path):
    f = run_dir / 'results_extended.json'
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text())
        m = data.get('test', {})
        return {
            'auc': m.get('auc'), 'f1': m.get('f1'), 'recall': m.get('recall'), 'precision': m.get('precision'),
            'composite': m.get('composite'), 'precision_aware': m.get('precision_aware')
        }
    except Exception:
        return None


def run_one(entry, common, npz, test_npz, epochs, seed, use_norm, balance, amplify, hn_factor, cosine, eta_min):
    args = [
        sys.executable, str(TRAIN), '--npz', str(npz), '--test_npz', str(test_npz), '--no_val',
        '--epochs', str(epochs), '--batch_size', str(entry['batch']), '--lr', str(entry['lr']), '--hard_negative_factor', str(hn_factor),
        '--gcn_hidden', str(common['gcn_hidden']), '--tcn_channels', str(common['tcn_channels']), '--tcn_kernel', str(common['tcn_kernel']),
        '--tcn_dropout', str(common['tcn_dropout']), '--tcn_dil_growth', str(common['tcn_dil_growth']), '--fc_hidden', str(common['fc_hidden']), '--fc_dropout', str(common['fc_dropout']),
        '--seed', str(seed)
    ]
    if use_norm: args.append('--use_norm')
    if balance: args.append('--balance_by_class')
    if amplify: args.append('--amplify_hard_negative')
    if cosine: args.extend(['--lr_schedule','cosine','--eta_min_factor',str(eta_min)])
    if entry['focal']>0:
        args.extend(['--focal_gamma', str(entry['focal'])])
    print('[RUN]', ' '.join(args))
    r = subprocess.run(args, cwd=str(THIS))
    if r.returncode!=0:
        print('[FAIL] batch', entry['batch'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=70)
    ap.add_argument('--seed', type=int, default=50)
    ap.add_argument('--skip-small', action='store_true', help='Skip batch=16 focal run')
    ap.add_argument('--npz', type=str, default=str(ROOT / 'train_data' / 'slipce_thresh040' / 'windows_dense_npz.npz'))
    ap.add_argument('--test-npz', type=str, default=str(ROOT / 'test_data' / 'slipce_thresh040' / 'windows_dense_npz.npz'))
    ap.add_argument('--use_norm', action='store_true')
    ap.add_argument('--balance', action='store_true')
    ap.add_argument('--amplify', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=1.5)
    ap.add_argument('--eta_min_factor', type=float, default=0.1)
    ap.add_argument('--sweep-after', action='store_true', help='Run threshold_sweep after all runs')
    args = ap.parse_args()

    # derive common params from a reference (optional: could read existing config) — here we just set typical values
    common = {
        'gcn_hidden':64,'tcn_channels':'128,128','tcn_kernel':3,'tcn_dropout':0.1,'tcn_dil_growth':2,'fc_hidden':128,'fc_dropout':0.2
    }
    plan = PLAN if not args.skip_small else [p for p in PLAN if p['batch']!=16]

    for entry in plan:
        run_one(entry, common, args.npz, args.test_npz, args.epochs, args.seed, args.use_norm, args.balance, args.amplify, args.hard_negative_factor, True, args.eta_min_factor)

    # collect
    rows=[]
    for d in sorted(RESULT.iterdir()):
        if d.is_dir() and (d/'config.json').exists():
            cfg=json.loads((d/'config.json').read_text())
            b=cfg.get('batch_size')
            if b not in [e['batch'] for e in plan]:
                continue
            metrics=parse_results(d)
            if not metrics: continue
            rows.append({'run':d.name,'batch_size':b,'lr':cfg.get('lr'),'focal_gamma':cfg.get('focal_gamma'),'lr_schedule':cfg.get('lr_schedule'), **metrics})
    if rows:
        out=THIS/'experiments_batch_refined.csv'
        headers=sorted({k for r in rows for k in r.keys()})
        with out.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(rows)
        print('[SUMMARY]', out)

    if args.sweep_after:
        sweep = THIS / 'threshold_sweep_gcn_tcn.py'
        if sweep.exists():
            subprocess.run([
                sys.executable, str(sweep), '--base', str(RESULT), '--test-npz', str(args.test_npz), '--min','0.05','--max','0.5','--step','0.01'
            ])

if __name__ == '__main__':
    main()
