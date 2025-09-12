#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.videoswin import build_videoswin3d_feature
from utils.preprocess import to_pseudo_video


def load_model(run_dir: Path, device: str, use_torchscript: bool=False, use_onnx: bool=False):
    if use_torchscript:
        ts_path = run_dir / 'model_ts.pt'
        m = torch.jit.load(str(ts_path), map_location=device).eval()
        return m, 'ts'
    if use_onnx:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            raise RuntimeError('需要 onnxruntime 以使用 ONNX 推論 (pip install onnxruntime)') from e
        avail = ort.get_available_providers()
        # 優先順序：CUDA > ROCM > DML > OpenVINO > CPU
        preferred = [
            'CUDAExecutionProvider',
            'ROCMExecutionProvider',
            'DmlExecutionProvider',
            'OpenVINOExecutionProvider',
            'CPUExecutionProvider'
        ]
        providers = [p for p in preferred if p in avail]
        if len(providers) == 0:
            providers = ['CPUExecutionProvider']
        sess = ort.InferenceSession(str(run_dir / 'model.onnx'), providers=providers)
        print(f"[INFO] ONNXRuntime providers used: {providers}")
        return sess, 'onnx'
    # fallback: pytorch state dict
    cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    mcfg = cfg['model']
    model = build_videoswin3d_feature(
        in_chans=mcfg['in_chans'], embed_dim=mcfg['embed_dim'], depths=mcfg['depths'], num_heads=mcfg['num_heads'],
        window_size=mcfg['window_size'], mlp_ratio=mcfg['mlp_ratio'], drop_rate=mcfg['drop_rate'], attn_drop_rate=mcfg['attn_drop_rate'],
        drop_path_rate=mcfg['drop_path_rate'], num_classes=mcfg['num_classes'], use_checkpoint=False).to(device)
    ckpt = torch.load(str(run_dir / 'best.ckpt'), map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def forward(self, frames):  # frames: (B,T,C,H,W)
            logits, _ = self.m(frames)  # train pipeline already gives (B,T,C,H,W)
            return logits
    return Wrapper(model), 'pt'


def infer(model_obj, kind: str, batch: torch.Tensor, device: str):
    if kind == 'onnx':
        import numpy as np
        ort_inputs = {'frames': batch.numpy()}
        logits = model_obj.run(['logits'], ort_inputs)[0]
        return torch.from_numpy(logits)
    else:
        batch = batch.to(device)
        if kind == 'ts':
            with torch.inference_mode():
                return model_obj(batch)
        else:  # pt
            with torch.inference_mode():
                return model_obj(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--input', required=True, help='Path to .npy or .npz containing (T,36) or batch list')
    ap.add_argument('--onnx', action='store_true')
    ap.add_argument('--torchscript', action='store_true')
    ap.add_argument('--topk', type=int, default=2)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_dir = Path(args.run_dir)

    # load sequences
    import numpy as np
    p = Path(args.input)
    data_list = []
    if p.suffix == '.npy':
        arr = np.load(p)
        if arr.ndim == 2:  # (T,F)
            data_list = [arr]
        elif arr.ndim == 3:  # (B,T,F)
            data_list = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError('npy shape must be (T,F) or (B,T,F)')
    elif p.suffix == '.npz':
        npz = np.load(p)
        # take all arrays inside
        for k in npz.files:
            arr = npz[k]
            if arr.ndim == 2:
                data_list.append(arr)
            elif arr.ndim == 3:
                data_list.extend([arr[i] for i in range(arr.shape[0])])
    else:
        raise ValueError('input must be .npy or .npz')

    batch = to_pseudo_video(data_list, feature_grid=(6,6), replicate_channels=3)  # (B,T,C,H,W)
    model_obj, kind = load_model(run_dir, device, use_torchscript=args.torchscript, use_onnx=args.onnx)
    logits = infer(model_obj, kind, batch, device)
    probs = torch.softmax(logits, dim=-1)
    topk = min(args.topk, probs.shape[-1])
    values, indices = torch.topk(probs, k=topk, dim=-1)
    for i in range(probs.shape[0]):
        print(f"Sample {i}: pred={indices[i,0].item()} prob={values[i,0].item():.4f} topk={list(zip(indices[i].tolist(), values[i].tolist()))}")

if __name__ == '__main__':
    main()
