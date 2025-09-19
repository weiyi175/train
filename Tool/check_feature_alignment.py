#!/usr/bin/env python3
"""Compare feature_list between training and test windows NPZ.
Usage:
  python Tool/check_feature_alignment.py \
      --train_npz train_data/slipce/windows_npz.npz \
      --test_npz  test_data/slipce/windows_npz.npz

It prints:
  - counts
  - missing in test
  - extra in test
  - recommended ordered intersection
  - reorder index mapping (test -> train) if partial overlap
Exit codes:
  0: identical
  1: differences found
"""
import argparse, sys, os, json
import numpy as np

def load_feat_list(path:str):
    if not os.path.isfile(path):
        print(f"[ERR] NPZ not found: {path}")
        sys.exit(2)
    D = np.load(path, allow_pickle=True)
    if 'feature_list' not in D:
        print(f"[ERR] feature_list missing in {path}")
        sys.exit(2)
    return list(D['feature_list'].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_npz', required=True)
    ap.add_argument('--test_npz', required=True)
    ap.add_argument('--save_report', help='Optional path to save JSON report')
    args = ap.parse_args()
    train_feat = load_feat_list(args.train_npz)
    test_feat = load_feat_list(args.test_npz)
    set_train = set(train_feat)
    set_test = set(test_feat)
    missing_in_test = [f for f in train_feat if f not in set_test]
    extra_in_test = [f for f in test_feat if f not in set_train]
    identical_order = (train_feat == test_feat)
    status = 'identical' if identical_order else 'diff'
    ordered_intersection = [f for f in train_feat if f in set_test]
    # mapping from test index -> train index (for potential re-ordering)
    mapping = {test_feat.index(f): train_feat.index(f) for f in ordered_intersection}
    report = {
        'train_count': len(train_feat),
        'test_count': len(test_feat),
        'identical_order': identical_order,
        'missing_in_test': missing_in_test,
        'extra_in_test': extra_in_test,
        'ordered_intersection': ordered_intersection,
        'test_to_train_index_map': mapping,
        'recommendation': None,
    }
    if identical_order:
        report['recommendation'] = 'OK: features identical and aligned.'
    else:
        if not missing_in_test and not extra_in_test:
            report['recommendation'] = 'Same set different order: reorder test tensors by test_to_train_index_map.'
        else:
            report['recommendation'] = 'Mismatch: consider regenerating test NPZ or restrict to ordered_intersection.'
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_report:
        with open(args.save_report,'w',encoding='utf-8') as f:
            json.dump(report,f,ensure_ascii=False,indent=2)
    sys.exit(0 if identical_order else 1)

if __name__ == '__main__':
    main()
