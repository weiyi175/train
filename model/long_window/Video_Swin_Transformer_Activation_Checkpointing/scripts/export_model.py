#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys, torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.videoswin import build_videoswin3d_feature
from datasets.smoke_dataset import build_dataloaders


def export(run_dir: str, onnx: bool, torchscript: bool, opset: int, dynamic: bool):
    run = Path(run_dir)
    cfg = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    mcfg = cfg['model']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_videoswin3d_feature(
        in_chans=mcfg['in_chans'], embed_dim=mcfg['embed_dim'], depths=mcfg['depths'], num_heads=mcfg['num_heads'],
        window_size=mcfg['window_size'], mlp_ratio=mcfg['mlp_ratio'], drop_rate=mcfg['drop_rate'], attn_drop_rate=mcfg['attn_drop_rate'],
        drop_path_rate=mcfg['drop_path_rate'], num_classes=mcfg['num_classes'], use_checkpoint=False).to(device)
    ckpt = torch.load(str(run / 'best.ckpt'), map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    # Get shape from dataset meta
    _, val_loader, meta = build_dataloaders(
        npz_path=cfg['data']['npz_path'], batch_size_micro=1, val_ratio=0.2, seed=cfg['training']['seed'],
        num_workers=0, balance_by_class=False, amplify_hard_negative=False, hard_negative_factor=1.0,
        temporal_jitter=0, feature_grid=tuple(cfg['data']['feature_grid']), replicate_channels=cfg['data']['replicate_channels'])
    T = meta['T']; C = meta['C']; H = meta['H']; W = meta['W']
    dummy = torch.randn(1, T, C, H, W, device=device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, x):  # x: (B,T,C,H,W)
            logits, feat = self.m(x.permute(0,1,2,3,4))
            return logits

    wrapper = Wrapper(model).to(device)

    if torchscript:
        ts_path = run / 'model_ts.pt'
        try:
            scripted = torch.jit.script(wrapper)
        except Exception as e:
            print(f"[WARN] torch.jit.script 失敗 ({e}); 改用 trace() 回退")
            scripted = torch.jit.trace(wrapper, dummy, strict=False)
        scripted.save(str(ts_path))
        print(f"[EXPORT] TorchScript saved: {ts_path}")

    if onnx:
        try:
            import onnx  # noqa: F401
        except Exception:
            print("[WARN] onnx 未安裝，請先 pip install onnx")
        onnx_path = run / 'model.onnx'
        dynamic_axes = {'frames': {0: 'batch'}}
        if dynamic:
            dynamic_axes['frames'][1] = 'time'
        torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=['frames'], output_names=['logits'],
                          opset_version=opset, dynamic_axes=dynamic_axes, do_constant_folding=True)
        print(f"[EXPORT] ONNX saved: {onnx_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--onnx', action='store_true')
    ap.add_argument('--torchscript', action='store_true')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--dynamic', action='store_true', help='Allow dynamic time length (experimental)')
    args = ap.parse_args()
    export(args.run_dir, args.onnx, args.torchscript, args.opset, args.dynamic)

if __name__ == '__main__':
    main()
