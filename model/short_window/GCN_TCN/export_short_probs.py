#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / 'Tool'
import sys as _sys
if str(TOOL) not in _sys.path:
    _sys.path.append(str(TOOL))
from dataset_npz import WindowsNPZDataset  # type: ignore
from models import GCN_TCN_Classifier  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='訓練輸出目錄 (包含 config.json, best.ckpt)')
    ap.add_argument('--npz', required=True, help='要推論的 npz (通常是 test)')
    ap.add_argument('--use-norm', action='store_true')
    ap.add_argument('--out-prefix', default='short')
    ap.add_argument('--batch', type=int, default=256)
    return ap.parse_args()


def load_model(run_dir: Path, device: str):
    cfg_path = run_dir / 'config.json'
    ckpt_path = run_dir / 'best.ckpt'
    if not cfg_path.exists() or not ckpt_path.exists():
        raise FileNotFoundError('缺少 config.json 或 best.ckpt')
    cfg = json.loads(cfg_path.read_text())
    params = cfg['params']
    in_dim = cfg['data']['F']
    model = GCN_TCN_Classifier(
        in_dim=in_dim, n_classes=2,
        gcn_hidden=params['gcn_hidden'],
        tcn_channels=tuple(int(x) for x in params['tcn_channels'].split(',')),
        tcn_kernel=params['tcn_kernel'], tcn_dropout=params['tcn_dropout'],
        tcn_dil_growth=params['tcn_dil_growth'],
        fc_hidden=params['fc_hidden'], fc_dropout=params['fc_dropout']
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, cfg


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = Path(args.run_dir)
    model, cfg = load_model(run_dir, device)
    ds = WindowsNPZDataset(npz_path=args.npz, split='short', use_norm=args.use_norm, temporal_jitter_frames=0)

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

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, collate_fn=collate)
    probs=[]; labels=[]
    with torch.no_grad():
        for batch in loader:
            x=batch['x'].to(device); m=batch['mask'].to(device); y=batch['y']
            logits,_=model(x,mask=m)
            p=torch.softmax(logits,dim=-1)[:,1].detach().cpu().numpy()
            probs.append(p); labels.append(y.numpy())
    probs=np.concatenate(probs,0)
    labels=np.concatenate(labels,0)
    np.save(run_dir / f"{args.out_prefix}_probs.npy", probs.astype('float32'))
    np.save(run_dir / f"{args.out_prefix}_labels.npy", labels.astype('int64'))
    print('Saved:', run_dir / f"{args.out_prefix}_probs.npy", run_dir / f"{args.out_prefix}_labels.npy")

if __name__=='__main__':
    main()
