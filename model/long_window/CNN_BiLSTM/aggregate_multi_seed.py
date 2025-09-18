#!/usr/bin/env python3
import os, json, glob, math, argparse
import pandas as pd
from pathlib import Path

# Compute metrics summary (mean, std, 95% CI) and correlations between
# validation checkpoint metrics and test metrics.

def mean_std_ci(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return (None, None, None, None)
    mean = s.mean(); std = s.std(ddof=1) if len(s) > 1 else 0.0
    n = len(s)
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return (mean, std, ci95, n)

def load_results(base_dir: Path):
    rows = []
    for seed_dir in sorted(base_dir.glob('seed*/')):
        # each result_dir uses next_result_dir pattern -> pick latest numeric subfolder
        subdirs = sorted([d for d in seed_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if not subdirs:
            continue
        latest = subdirs[-1]
        rj = latest / 'results.json'
        if not rj.exists():
            continue
        try:
            data = json.loads(rj.read_text(encoding='utf-8'))
        except Exception as e:
            print('[WARN] read fail', rj, e); continue
        test = data.get('test', {})
        params = data.get('params', {})
        # Derive best validation composite / precision-aware
        top_comp = data.get('top_epochs', [])
        top_prec = data.get('top_epochs_precision_aware', [])
        row = {
            'seed': params.get('run_seed'),
            'test_auc': test.get('auc'),
            'test_f1': test.get('f1'),
            'test_recall': test.get('recall'),
            'test_precision': test.get('precision'),
            'test_composite': test.get('composite'),
            'test_precision_aware': test.get('precision_aware'),
            'val_best_composite': top_comp[0]['score'] if top_comp else None,
            'val_best_precision_aware': top_prec[0]['precision_aware'] if top_prec else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)

def correlations(df: pd.DataFrame):
    out = {}
    pairs = [
        ('val_best_composite','test_composite'),
        ('val_best_composite','test_precision_aware'),
        ('val_best_precision_aware','test_precision_aware'),
        ('val_best_precision_aware','test_composite'),
    ]
    for a,b in pairs:
        if a in df and b in df and df[a].notna().any() and df[b].notna().any():
            try:
                out[f'{a}__{b}'] = df[[a,b]].dropna().corr().iloc[0,1]
            except Exception:
                out[f'{a}__{b}'] = None
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='result_multi_seed_precision_aware', help='Base multi-seed directory (run_multi_seed.sh output)')
    ap.add_argument('--out', default=None, help='Optional summary CSV output path')
    args = ap.parse_args()

    base_dir = Path(args.base)
    if not base_dir.exists():
        raise SystemExit(f'Base directory not found: {base_dir}')

    df = load_results(base_dir)
    if df.empty:
        raise SystemExit('No seed results found under ' + str(base_dir))

    metrics = ['test_auc','test_f1','test_recall','test_precision','test_composite','test_precision_aware']
    summary_rows = []
    for m in metrics:
        mean, std, ci95, n = mean_std_ci(df[m])
        summary_rows.append({'metric': m, 'mean': mean, 'std': std, 'ci95': ci95, 'n': n})
    summary_df = pd.DataFrame(summary_rows)

    corr = correlations(df)

    print('=== Raw per-seed ===')
    print(df.to_string(index=False))
    print('\n=== Summary (mean/std/95%CI) ===')
    print(summary_df.to_string(index=False))
    print('\n=== Validation/Test Correlations ===')
    for k,v in corr.items():
        print(f'{k}: {v:.4f}' if v is not None else f'{k}: NA')

    if args.out:
        summary_csv = Path(args.out)
        summary_df.to_csv(summary_csv, index=False)
        print('Saved summary ->', summary_csv)
        corr_path = summary_csv.parent / (summary_csv.stem + '_corr.json')
        import json as _j; corr_path.write_text(_j.dumps(corr, indent=2))
        print('Saved correlations ->', corr_path)

if __name__ == '__main__':
    main()
