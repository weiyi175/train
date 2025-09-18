#!/usr/bin/env python3
import json, sys, os, csv
from pathlib import Path


# Unified scoring formulas
# composite: 0.5*Recall + 0.3*F1 + 0.2*AUC
# precision_aware: 0.5*Precision + 0.3*F1 + 0.2*AUC


def composite(auc, f1, recall):
    try:
        return 0.5*float(recall) + 0.3*float(f1) + 0.2*float(auc)
    except Exception:
        return None


def precision_aware(auc, f1, precision):
    try:
        return 0.5*float(precision) + 0.3*float(f1) + 0.2*float(auc)
    except Exception:
        return None


def iter_results(sweep_dir: Path):
    for p in sweep_dir.rglob('results.json'):
        if sweep_dir.as_posix() not in p.as_posix():
            continue
        yield p


def load_json(p: Path):
    try:
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print('WARN read', p, e)
        return None


def fmt(v):
    return 'NA' if v is None else f"{v:.4f}" if isinstance(v,(int,float)) else str(v)


def main():
    base = Path(__file__).parent
    sweep_dir = base / 'result_mask_sweep'
    if not sweep_dir.exists():
        print('No sweep dir found:', sweep_dir)
        sys.exit(1)
    rows = []
    for rp in sorted(iter_results(sweep_dir)):
        data = load_json(rp)
        if not data:
            continue
        params = data.get('params', {})
        test = data.get('test', {})
        tops = data.get('top_epochs', [])
        top_prec = data.get('top_epochs_precision_aware', [])
        auc = test.get('auc'); f1 = test.get('f1'); recall = test.get('recall')
        precision = test.get('precision')
        tp = test.get('TP'); fp = test.get('FP'); fn = test.get('FN'); tn = test.get('TN')
        comp = test.get('composite'); prec_aware = test.get('precision_aware')
        if comp is None:
            comp = composite(auc, f1, recall)
        if prec_aware is None and precision is not None:
            prec_aware = precision_aware(auc, f1, precision)
        best_epoch = tops[0]['epoch'] if tops else None
        best_comp = tops[0].get('score') if tops else None
        best_prec_epoch = top_prec[0]['epoch'] if top_prec else None
        best_prec_score = top_prec[0].get('precision_aware') if top_prec else None
        rows.append({
            'run_dir': rp.parent.as_posix(),
            'threshold': params.get('mask_threshold'),
            'window_gate': params.get('window_mask_min_mean'),
            'effective_batch': params.get('effective_batch'),
            'seed': params.get('run_seed'),
            'test_auc': auc,
            'test_f1': f1,
            'test_recall': recall,
            'test_precision': precision,
            'test_composite': comp,
            'test_precision_aware': prec_aware,
            'test_TP': tp, 'test_FP': fp, 'test_FN': fn, 'test_TN': tn,
            'val_best_epoch_composite': best_epoch,
            'val_best_composite': best_comp,
            'val_best_epoch_precision_aware': best_prec_epoch,
            'val_best_precision_aware': best_prec_score,
        })
    if not rows:
        print('No results found under', sweep_dir)
        sys.exit(2)
    out_csv = sweep_dir / 'aggregate.csv'
    headers = list(rows[0].keys())
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerows(rows)
    top_comp = sorted(rows, key=lambda r: (r['test_composite'] or -1), reverse=True)[:5]
    top_prec = sorted(rows, key=lambda r: (r['test_precision_aware'] or -1), reverse=True)[:5]
    print(f'Wrote {len(rows)} rows -> {out_csv}')
    print('Top 5 by test_composite:')
    for r in top_comp:
        print(f"th={r['threshold']} gate={r['window_gate']} comp={fmt(r['test_composite'])} auc={fmt(r['test_auc'])} f1={fmt(r['test_f1'])} recall={fmt(r['test_recall'])} prec={fmt(r['test_precision'])} -> {r['run_dir']}")
    print('Top 5 by test_precision_aware:')
    for r in top_prec:
        print(f"th={r['threshold']} gate={r['window_gate']} pAware={fmt(r['test_precision_aware'])} auc={fmt(r['test_auc'])} f1={fmt(r['test_f1'])} recall={fmt(r['test_recall'])} prec={fmt(r['test_precision'])} -> {r['run_dir']}")


if __name__ == '__main__':
    main()
