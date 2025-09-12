#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from datasets.smoke_dataset import build_dataloaders
from models.videoswin import build_videoswin3d_feature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _, val_loader, _ = build_dataloaders(
        npz_path=cfg['data']['npz_path'], batch_size_micro=cfg['data']['batch_size_micro'], val_ratio=0.2,
        seed=cfg['training']['seed'], num_workers=cfg['data']['num_workers'], balance_by_class=False,
        amplify_hard_negative=False, hard_negative_factor=1.0, temporal_jitter=0,
        feature_grid=tuple(cfg['data']['feature_grid']), replicate_channels=cfg['data']['replicate_channels'])
    mcfg = cfg['model']
    model = build_videoswin3d_feature(
        in_chans=mcfg['in_chans'], embed_dim=mcfg['embed_dim'], depths=mcfg['depths'], num_heads=mcfg['num_heads'],
        window_size=mcfg['window_size'], mlp_ratio=mcfg['mlp_ratio'], drop_rate=mcfg['drop_rate'], attn_drop_rate=mcfg['attn_drop_rate'],
        drop_path_rate=mcfg['drop_path_rate'], num_classes=mcfg['num_classes'], use_checkpoint=False).to(device)
    ckpt = torch.load(str(run / 'best.ckpt'), map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    total_correct=0; total_n=0
    with torch.no_grad():
        for batch in val_loader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            logits,_ = model(frames.permute(0,1,2,3,4))
            pred = logits.argmax(-1)
            total_correct += (pred==labels).sum().item()
            total_n += labels.size(0)
    acc = total_correct / max(1,total_n)
    print(json.dumps({'val_acc': acc, 'samples': total_n}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
