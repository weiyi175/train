#!/usr/bin/env python3
from __future__ import annotations
"""Inference script for Video Swin 3D runs.

Usage:
  python scripts/infer_videoswin.py --run_dir path/to/result_mod/10 --input_npz windows_v2_all.npz [--threshold 0.5]
If --threshold not provided, tries to load best_threshold.txt inside run_dir;
if still missing, falls back to 0.5.

Can also auto-load inference_config.json if present to override model meta.
"""
import argparse, json
from pathlib import Path
import sys, torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from models.videoswin import build_videoswin3d_preset
from datasets.smoke_dataset import SmokeLongWindowPseudoVideo

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True, help='Training run directory containing best.ckpt / inference_config.json')
    ap.add_argument('--input_npz', required=True, help='NPZ path for evaluation')
    ap.add_argument('--threshold', type=float, default=None, help='Override threshold (otherwise auto)')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--val_ratio', type=float, default=0.0, help='Ignored (we only evaluate entire set)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    return ap.parse_args()

def load_threshold(run_dir: Path, override: float|None):
    if override is not None:
        return override, 'override'
    txt = run_dir / 'best_threshold.txt'
    if txt.exists():
        try:
            val = float(txt.read_text().strip())
            return val, 'best_threshold.txt'
        except Exception:
            pass
    return 0.5, 'default_0.5'

def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f'[error] run_dir not found: {run_dir}')
        return
    infer_cfg_path = run_dir / 'inference_config.json'
    infer_cfg = None
    if infer_cfg_path.exists():
        try:
            infer_cfg = json.loads(infer_cfg_path.read_text())
            print(f'[info] Loaded inference_config.json')
        except Exception as e:
            print('[warn] Failed to parse inference_config.json:', e)
    # Determine model params
    preset = infer_cfg.get('preset') if infer_cfg else 'tiny'
    window_size = tuple(infer_cfg.get('window_size', (2,7,7))) if infer_cfg else (2,7,7)
    single_logit = infer_cfg.get('single_logit', True) if infer_cfg else True
    in_chans = infer_cfg.get('in_chans', 3) if infer_cfg else 3
    num_classes = 1 if single_logit else 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Build dataset with proper feature pack components if enabled
    fp_cfg = (infer_cfg.get('feature_pack') if infer_cfg else {}) or {}
    use_fp = bool(fp_cfg.get('enabled'))
    comps = fp_cfg.get('components') or {}
    fp_velocity = comps.get('velocity', True)
    fp_accel = comps.get('accel', True)
    fp_energy = comps.get('energy', True)
    fp_pairwise = comps.get('pairwise', False)
    fp_joints = fp_cfg.get('joints', 15)
    fp_dims_per_joint = fp_cfg.get('dims_per_joint', 4)
    fp_pairwise_subset = fp_cfg.get('pairwise_subset', 20)
    ds = SmokeLongWindowPseudoVideo(
        args.input_npz, split='long', use_norm=True, temporal_jitter=0, feature_grid=(6,6),
        replicate_channels=in_chans if not use_fp else 3,
        use_feature_pack=use_fp,
        fp_velocity=fp_velocity, fp_accel=fp_accel, fp_energy=fp_energy, fp_pairwise=fp_pairwise,
        fp_joints=fp_joints, fp_dims_per_joint=fp_dims_per_joint, fp_pairwise_subset=fp_pairwise_subset
    )
    from torch.utils.data import DataLoader
    def collate(batch):
        import torch
        frames = torch.stack([b['frames'] for b in batch],0)
        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)
        return {'frames':frames,'label':labels}
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    # If in_chans unspecified, infer from dataset
    if in_chans is None:
        sample = ds[0]['frames']
        in_chans = sample.shape[1]
    # Build model
    model = build_videoswin3d_preset(preset, in_chans=in_chans, num_classes=num_classes, window_size=window_size, use_checkpoint=False).to(device)
    ckpt_path = run_dir / 'best.ckpt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'last.ckpt'
    if not ckpt_path.exists():
        print('[error] no checkpoint found in run_dir')
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    # Threshold
    threshold, th_src = load_threshold(run_dir, args.threshold)
    print(f'[threshold] using {threshold} (source={th_src})')
    ys=[]; ps=[]; probs=[]
    import torch.nn.functional as F
    with torch.no_grad():
        for b in loader:
            x=b['frames'].to(device)
            logits,_=model(x)
            if single_logit:
                prob = torch.sigmoid(logits.squeeze(1))
                pred = (prob>=threshold).long()
            else:
                prob_full = F.softmax(logits, dim=-1)
                prob = prob_full[:,1]
                pred = (prob>=threshold).long()
            ys.append(b['label']); ps.append(pred.cpu()); probs.append(prob.cpu())
    y = torch.cat(ys); p = torch.cat(ps); pr = torch.cat(probs)
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
    f1 = f1_score(y.numpy(), p.numpy())
    try:
        auc = roc_auc_score(y.numpy(), pr.numpy())
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y.numpy(), p.numpy())
    acc = (p==y).float().mean().item()
    print(f'[metrics] f1={f1:.4f} auc={auc:.4f} acc={acc:.4f} threshold={threshold}')
    print(f'[cm]\n{cm}')

if __name__=='__main__':
    main()
