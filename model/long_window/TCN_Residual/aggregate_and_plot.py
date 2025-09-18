#!/usr/bin/env python3
import json
import glob
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__)
RESULT_DIR = os.path.join(ROOT, 'result')
PLOTS_DIR = os.path.join(RESULT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

records = []
for fp in sorted(glob.glob(os.path.join(RESULT_DIR, '*/results.json'))):
    run = os.path.basename(os.path.dirname(fp))
    try:
        with open(fp, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print('skip', fp, 'err', e)
        continue
    test = data.get('test', {})
    params = data.get('params', {})
    cmd = data.get('cmd')
    top = data.get('top_epochs', [])
    # top epochs: keep top1 score
    top1 = top[0] if top else {}
    records.append({
        'run': run,
        'auc': test.get('auc'),
        'f1': test.get('f1'),
        'recall': test.get('recall'),
        'TP': test.get('TP'),
        'FP': test.get('FP'),
        'FN': test.get('FN'),
        'TN': test.get('TN'),
        'param_gate': params.get('gate_type'),
        'param_attn_units': params.get('attn_units'),
        'cmd': cmd,
        'top1_epoch': top1.get('epoch'),
        'top1_score': top1.get('score')
    })

if not records:
    print('no records found')
    raise SystemExit(1)

df = pd.DataFrame(records)
df_sorted = df.sort_values('run')
agg_fp = os.path.join(RESULT_DIR, 'aggregate.csv')
df_sorted.to_csv(agg_fp, index=False)
print('wrote', agg_fp)

# Plot distributions
metrics = ['auc', 'f1', 'recall']
for m in metrics:
    plt.figure(figsize=(6,4))
    plt.hist(df_sorted[m].dropna(), bins=12, alpha=0.9)
    plt.title(f'Distribution of {m}')
    plt.xlabel(m)
    plt.ylabel('count')
    pfp = os.path.join(PLOTS_DIR, f'{m}_hist.png')
    plt.tight_layout()
    plt.savefig(pfp)
    plt.close()
    print('wrote', pfp)

# Boxplot
plt.figure(figsize=(6,4))
df_sorted[['auc','f1','recall']].boxplot()
plt.title('Boxplot AUC / F1 / Recall')
bfp = os.path.join(PLOTS_DIR, 'boxplot_metrics.png')
plt.tight_layout()
plt.savefig(bfp)
plt.close()
print('wrote', bfp)

# Scatter AUC vs F1
plt.figure(figsize=(6,5))
plt.scatter(df_sorted['auc'], df_sorted['f1'])
for i,r in df_sorted.iterrows():
    plt.text(r['auc'], r['f1'], r['run'], fontsize=6)
plt.xlabel('AUC')
plt.ylabel('F1')
plt.title('AUC vs F1 (runs labeled)')
sp = os.path.join(PLOTS_DIR, 'auc_vs_f1.png')
plt.tight_layout()
plt.savefig(sp)
plt.close()
print('wrote', sp)

print('\nSummary:')
print(df_sorted[['run','auc','f1','recall','TP','FP','FN','TN','top1_epoch','top1_score']].to_string(index=False))
