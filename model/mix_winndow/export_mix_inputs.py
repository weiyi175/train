#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
from pathlib import Path
import numpy as np
import torch

# Reuse short-window inference utils
from pathlib import Path as _P
import sys as _sys
ROOT = _P(__file__).resolve().parents[2]
TOOL = ROOT / 'Tool'
SHORT_DIR = ROOT / 'model' / 'short_window' / 'GCN_TCN'
LONG_DIR = ROOT / 'model' / 'long_window' / 'CNN_BiLSTM'
for p in [TOOL, SHORT_DIR, LONG_DIR]:
    if str(p) not in _sys.path:
        _sys.path.append(str(p))
from dataset_npz import WindowsNPZDataset  # type: ignore

# Short-window model loader (absolute import after path patch)
from export_short_probs import load_model as load_short_model  # type: ignore

# Long-window utils (to get labels consistently)
import importlib.machinery, importlib.util
_long_utils_path = LONG_DIR / 'utils.py'
_loader = importlib.machinery.SourceFileLoader('cnn_long_utils', str(_long_utils_path))
_spec = importlib.util.spec_from_loader(_loader.name, _loader)
_mod = importlib.util.module_from_spec(_spec)
_loader.exec_module(_mod)  # type: ignore
load_npz_windows = getattr(_mod, 'load_npz_windows')  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--short-run', required=True, help='短視窗 run 目錄 (含 best.ckpt)')
    ap.add_argument('--short-npz', required=True, help='短視窗對應 npz (test)')
    ap.add_argument('--short-use-norm', action='store_true')
    ap.add_argument('--long-npz', required=True, help='長視窗對應 npz (test)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--prefix', default='mix')
    return ap.parse_args()


def export_short(short_run: Path, short_npz: str, use_norm: bool, out_dir: Path, prefix: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = load_short_model(short_run, device)
    ds = WindowsNPZDataset(npz_path=short_npz, split='short', use_norm=use_norm, temporal_jitter_frames=0)
    from torch.utils.data import DataLoader
    import numpy as np
    def collate(batch):
        xs,masks,ys=[],[],[]
        for b in batch:
            x=b['x']; m=b.get('mask'); y=b['y']
            if isinstance(x,np.ndarray): x=torch.from_numpy(x)
            if isinstance(m,np.ndarray): m=torch.from_numpy(m)
            if m is None: m_time=torch.ones(x.shape[0],dtype=torch.bool)
            else:
                mb=m.bool(); m_time=mb.any(dim=-1) if mb.ndim==2 else mb
            xs.append(torch.nan_to_num(x,nan=0.0)); masks.append(m_time); ys.append(int(y))
        return {'x':torch.stack(xs,0),'mask':torch.stack(masks,0),'y':torch.tensor(ys)}
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate)
    probs=[]
    with torch.no_grad():
        for batch in loader:
            x=batch['x'].to(device); m=batch['mask'].to(device)
            logits,_=model(x,mask=m)
            p=torch.softmax(logits,dim=-1)[:,1].detach().cpu().numpy()
            probs.append(p)
    probs=np.concatenate(probs,0)
    np.save(out_dir / f'{prefix}_short_probs.npy', probs.astype('float32'))


def export_long_labels(long_npz: str, out_dir: Path, prefix: str):
    X, y = load_npz_windows(long_npz)
    # y could be shape (N,) or (N,1) -> squeeze
    y = np.asarray(y).reshape(-1).astype('int64')
    np.save(out_dir / f'{prefix}_labels.npy', y)


# NOTE: 長視窗機率輸出：這裡先從已存在的集成輸出或另行提供 long_probs.npy。
# 若需要從 Keras 權重直接推論，需補一個 TF 推論腳本；先留接口：

def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) short probs
    export_short(Path(args.short_run), args.short_npz, args.short_use_norm, out_dir, args.prefix)

    # 2) labels from long npz (確保與 long 資料一致)
    export_long_labels(args.long_npz, out_dir, args.prefix)

    print('Done. Files:')
    for n in [f'{args.prefix}_short_probs.npy', f'{args.prefix}_labels.npy']:
        p = out_dir / n
        print('-', p, p.exists(), os.path.getsize(p) if p.exists() else 0)

if __name__=='__main__':
    main()
