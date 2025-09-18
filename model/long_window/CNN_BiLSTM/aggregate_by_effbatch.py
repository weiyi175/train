#!/usr/bin/env python3
import os, json
import pandas as pd

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, 'result')
OUT = os.path.join(RES, 'effbatch_stats.csv')

rows = []
for name in sorted(os.listdir(RES)):
	d = os.path.join(RES, name)
	f = os.path.join(d, 'results.json')
	if os.path.isdir(d) and os.path.exists(f):
		try:
			obj = json.load(open(f, 'r', encoding='utf-8'))
			p = obj.get('params', {})
			t = obj.get('test', {})
			rows.append({
				'run': name,
				'auc': t.get('auc'), 'f1': t.get('f1'), 'recall': t.get('recall'),
				'effective_batch': p.get('effective_batch'),
				'batch': p.get('batch'), 'accumulate_steps': p.get('accumulate_steps'),
				'seed': p.get('run_seed'),
			})
		except Exception:
			pass

if not rows:
	raise SystemExit('No results found')

df = pd.DataFrame(rows)
df['score'] = 0.5*df['recall'] + 0.3*df['f1'] + 0.2*df['auc']

stat = df.groupby(['effective_batch']).agg(
	count=('run','count'),
	auc_mean=('auc','mean'), auc_std=('auc','std'),
	f1_mean=('f1','mean'), f1_std=('f1','std'),
	recall_mean=('recall','mean'), recall_std=('recall','std'),
	score_mean=('score','mean'), score_std=('score','std'),
).reset_index().sort_values('score_mean', ascending=False)

os.makedirs(RES, exist_ok=True)
stat.to_csv(OUT, index=False)
print('Saved ->', OUT)
print(stat.to_string(index=False))

