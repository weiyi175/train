#!/usr/bin/env python3
import os, json, glob
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), 'result')

rows = []
for d in sorted(glob.glob(os.path.join(BASE, '*'))):
    rj = os.path.join(d, 'results.json')
    if not os.path.isfile(rj):
        continue
    try:
        with open(rj,'r',encoding='utf-8') as f:
            r = json.load(f)
        test = r.get('test', {})
        params = r.get('params', {})
        precision = test.get('precision')
        precision_aware = test.get('precision_aware')
        row = {
            'run': os.path.basename(d),
            'auc': test.get('auc'), 'f1': test.get('f1'), 'recall': test.get('recall'), 'precision': precision,
            'TP': test.get('TP'), 'FP': test.get('FP'), 'FN': test.get('FN'), 'TN': test.get('TN'),
            'batch': params.get('batch'), 'accumulate_steps': params.get('accumulate_steps'), 'effective_batch': params.get('effective_batch'),
            'composite': test.get('composite'), 'precision_aware': precision_aware,
        }
        rows.append(row)
    except Exception as e:
        print('[WARN] failed to read', rj, '->', e)

if not rows:
    raise SystemExit('No results found under ' + BASE)

df = pd.DataFrame(rows)
# Recompute in case older results lack fields
if 'composite' not in df or df['composite'].isna().any():
    df['composite'] = 0.5*df['recall'] + 0.3*df['f1'] + 0.2*df['auc']
if 'precision_aware' not in df or df['precision_aware'].isna().any():
    df['precision_aware'] = 0.5*df['precision'].fillna(0) + 0.3*df['f1'] + 0.2*df['auc']

# Sort primarily by effective_batch then composite desc
df = df.sort_values(['effective_batch','composite'], ascending=[True, False])

out_csv = os.path.join(BASE, 'aggregate_cnn_bilstm.csv')
df.to_csv(out_csv, index=False)
print('Saved ->', out_csv)
print(df[['run','effective_batch','auc','precision','recall','f1','composite','precision_aware']].to_string(index=False))
