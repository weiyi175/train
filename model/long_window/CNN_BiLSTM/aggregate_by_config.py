#!/usr/bin/env python3
import os
import json
import math
import pandas as pd

HERE = os.path.dirname(__file__)
RES_DIR = os.path.join(HERE, 'result')
OUT_CSV = os.path.join(RES_DIR, 'config_stats.csv')

rows = []
for name in sorted(os.listdir(RES_DIR)):
    path = os.path.join(RES_DIR, name, 'results.json')
    if os.path.isdir(os.path.join(RES_DIR, name)) and os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            p = obj.get('params', {})
            t = obj.get('test', {})
            rows.append({
                'run': name,
                'auc': t.get('auc'),
                'f1': t.get('f1'),
                'recall': t.get('recall'),
                'TP': t.get('TP'), 'FP': t.get('FP'), 'FN': t.get('FN'), 'TN': t.get('TN'),
                'batch': p.get('batch'),
                'accumulate_steps': p.get('accumulate_steps'),
                'effective_batch': p.get('effective_batch'),
                'seed': p.get('run_seed'),
            })
        except Exception:
            pass

if not rows:
    raise SystemExit('No results found under result/*/results.json')

df = pd.DataFrame(rows)
# compute composite score consistent with training callback
score = 0.5*df['recall'] + 0.3*df['f1'] + 0.2*df['auc']
df['score'] = score

# group by config
grp_cols = ['batch', 'accumulate_steps', 'effective_batch']
aggs = df.groupby(grp_cols).agg(
    count=('run', 'count'),
    auc_mean=('auc', 'mean'), auc_std=('auc', 'std'),
    f1_mean=('f1', 'mean'), f1_std=('f1', 'std'),
    recall_mean=('recall', 'mean'), recall_std=('recall', 'std'),
    score_mean=('score', 'mean'), score_std=('score', 'std'),
).reset_index().sort_values('score_mean', ascending=False)

os.makedirs(RES_DIR, exist_ok=True)
aggs.to_csv(OUT_CSV, index=False)
print('Saved ->', OUT_CSV)
print(aggs.to_string(index=False))
