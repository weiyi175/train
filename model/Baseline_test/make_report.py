#!/usr/bin/env python3
from __future__ import annotations
import os, sys, argparse

# 路徑：使得相對匯入可行
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.Baseline_test.train_baseline import generate_run_report  # type: ignore


def find_run_dirs(base: str):
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and name.isdigit():
            # has required files
            if os.path.exists(os.path.join(p, 'train_log.jsonl')):
                out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description='Generate detailed reports for Baseline_test results')
    ap.add_argument('--result_dir', default=os.path.join(ROOT, 'model', 'Baseline_test', 'result'))
    ap.add_argument('--run_dir', action='append', help='specific run directory to process; can repeat')
    ap.add_argument('--all', action='store_true', help='process all runs under result_dir')
    args = ap.parse_args()

    run_dirs = []
    if args.run_dir:
        run_dirs.extend(args.run_dir)
    if args.all or not run_dirs:
        run_dirs.extend(find_run_dirs(args.result_dir))

    if not run_dirs:
        print('No run directories found to process.')
        return

    for rd in run_dirs:
        try:
            generate_run_report(rd)
            print(f"Generated report for: {rd}")
        except Exception as e:
            print(f"Failed to generate report for {rd}: {e}")


if __name__ == '__main__':
    main()
