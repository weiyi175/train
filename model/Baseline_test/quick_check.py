#!/usr/bin/env python3
from __future__ import annotations
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TOOL_DIR = os.path.join(ROOT, 'Tool')
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)
from dataset_npz import WindowsNPZDataset

def main():
    npz = os.path.join(ROOT, 'train_data', 'slipce', 'windows_dense_npz.npz')
    ds = WindowsNPZDataset(npz, split='short', use_norm=True)
    print('short:', ds.x.shape, 'labels:', int(ds.y.sum()), '/', len(ds))
    ds2 = WindowsNPZDataset(npz, split='long', use_norm=True)
    print('long :', ds2.x.shape, 'labels:', int(ds2.y.sum()), '/', len(ds2))

if __name__ == '__main__':
    main()
