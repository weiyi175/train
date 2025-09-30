#!/usr/bin/env python3
"""Sweep batch sizes for GCN_TCN training.

Creates a subfolder runs_batch/<batch_size>/ for each tested batch size, reusing core hyperparameters
from a reference config (e.g. an existing run like result/22). Uses a reduced epoch count for quick
comparative diagnostics unless --full flag is set.

Outputs: runs_batch/summary_batch_size.csv with metrics at threshold=0.5.
Optionally can also trigger threshold sweep per run (flag --sweep-threshold) after training.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, os, csv, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).parent
RESULT_DIR = THIS_DIR / 'result'

DEFAULT_BATCHES = [16,32,48,64,128,256]

def load_reference_config(ref_run: str|None) -> dict:
    if not ref_run:
        return {}
    cfg_path = RESULT_DIR / ref_run / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'reference config not found: {cfg_path}')
    return json.loads(cfg_path.read_text(encoding='utf-8'))


def build_command(args, batch_size: int, ref: dict, out_dir: Path) -> list[str]:
    base_npz = args.npz or (ROOT / 'train_data' / 'slipce_thresh040' / 'windows_dense_npz.npz')
    test_npz = args.test_npz or (ROOT / 'test_data' / 'slipce_thresh040' / 'windows_dense_npz.npz')
    epochs = args.epochs if args.epochs else min( ref.get('epochs', 30), 5 if not args.full else ref.get('epochs',30) )
    lr = args.lr if args.lr else ref.get('lr', 1e-4)
    seed = ref.get('seed', 42)
    use_norm = '--use_norm' if ref.get('use_norm', False) else ''
    balance = '--balance_by_class' if ref.get('balance_by_class', False) else ''
    amplify = '--amplify_hard_negative' if ref.get('amplify_hard_negative', False) else ''
    hn_factor = ref.get('hard_negative_factor', 1.5)
    params = ref.get('params', {})
    gcn_hidden = params.get('gcn_hidden', 64)
    tcn_channels = params.get('tcn_channels', '128,128')
    tcn_kernel = params.get('tcn_kernel', 3)
    tcn_dropout = params.get('tcn_dropout', 0.1)
    tcn_dil_growth = params.get('tcn_dil_growth',2)
    fc_hidden = params.get('fc_hidden', 128)
    fc_dropout = params.get('fc_dropout', 0.2)

    cmd = [
        sys.executable, str(THIS_DIR / 'train_gcn_tcn.py'),
        '--npz', str(base_npz), '--test_npz', str(test_npz), '--no_val',
        '--epochs', str(epochs), '--batch_size', str(batch_size), '--lr', str(lr),
        '--hard_negative_factor', str(hn_factor), '--gcn_hidden', str(gcn_hidden),
        '--tcn_channels', str(tcn_channels), '--tcn_kernel', str(tcn_kernel), '--tcn_dropout', str(tcn_dropout),
        '--tcn_dil_growth', str(tcn_dil_growth), '--fc_hidden', str(fc_hidden), '--fc_dropout', str(fc_dropout),
        '--seed', str(seed)
    ]
    for flag in [use_norm, balance, amplify]:
        if flag:
            cmd.append(flag)
    return cmd


def parse_results(run_dir: Path) -> dict|None:
    f = run_dir / 'results_extended.json'
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding='utf-8'))
        m = data.get('test', {})
        return {
            'auc': m.get('auc'), 'f1': m.get('f1'), 'recall': m.get('recall'), 'precision': m.get('precision'),
            'composite': m.get('composite'), 'precision_aware': m.get('precision_aware')
        }
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-run', type=str, default='22', help='Reference run id under result/ for hyperparams')
    ap.add_argument('--batches', type=str, default=','.join(str(b) for b in DEFAULT_BATCHES))
    ap.add_argument('--epochs', type=int, default=0, help='Override epochs (0=auto)')
    ap.add_argument('--full', action='store_true', help='Use full reference epochs instead of quick mode')
    ap.add_argument('--npz', type=str, default='')
    ap.add_argument('--test-npz', type=str, default='')
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--sweep-threshold', action='store_true', help='Run threshold sweep after each training run')
    ap.add_argument('--force-retrain', action='store_true', help='If run folder exists, retrain (delete and recreate)')
    args = ap.parse_args()

    ref_cfg = load_reference_config(args.ref_run) if args.ref_run else {}
    batches = [int(x) for x in args.batches.split(',') if x.strip()]

    out_root = THIS_DIR / 'runs_batch'
    out_root.mkdir(exist_ok=True)

    rows=[]
    for b in batches:
        tag = f'b{b}'
        run_dir = out_root / tag
        if run_dir.exists() and args.force_retrain:
            shutil.rmtree(run_dir)
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
        # set environment to force next run_dir allocation inside train script into our custom path?
        # Instead: after training copy produced run folder into this tag directory (simpler) because train_gcn_tcn auto increments.
        before = set(p for p in RESULT_DIR.iterdir() if p.is_dir())
        cmd = build_command(args, b, ref_cfg, run_dir)
        print('[RUN]', ' '.join(cmd))
        r = subprocess.run(cmd, cwd=str(THIS_DIR))
        if r.returncode != 0:
            print(f'[FAIL] batch={b} returncode={r.returncode}')
            continue
        after = set(p for p in RESULT_DIR.iterdir() if p.is_dir())
        new_dirs = sorted(list(after - before), key=lambda p: p.stat().st_mtime)
        if not new_dirs:
            print(f'[WARN] cannot identify new run dir for batch {b}')
            continue
        latest = new_dirs[-1]
        # copy artifacts into run_dir (overwrite minimal)
        for fname in ['config.json','results_extended.json','train_log.jsonl','report.md','last.ckpt','best.ckpt']:
            src = latest / fname
            if src.exists():
                shutil.copy2(src, run_dir / fname)
        metrics = parse_results(run_dir)
        row={'batch_size': b}
        if metrics:
            row.update(metrics)
        rows.append(row)

        if args.sweep_threshold:
            sweep_script = THIS_DIR / 'threshold_sweep_gcn_tcn.py'
            if sweep_script.exists():
                # run sweep just for this run (simulate by pointing --base to its parent and filtering later if needed)
                subprocess.run([
                    sys.executable, str(sweep_script), '--base', str(RESULT_DIR), '--test-npz', args.test_npz or str(ROOT / 'test_data' / 'slipce_thresh040' / 'windows_dense_npz.npz'), '--min','0.05','--max','0.5','--step','0.01'
                ])

    # write summary
    if rows:
        summary_path = out_root / 'summary_batch_size.csv'
        headers = sorted({k for r in rows for k in r.keys()})
        with summary_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader(); w.writerows(rows)
        print('[SUMMARY]', summary_path)
    else:
        print('[WARN] no successful runs recorded.')

if __name__ == '__main__':
    main()
