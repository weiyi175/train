#!/usr/bin/env python3
from __future__ import annotations
"""Capacity Probe Utility
快速探測在目前 GPU/設定下可承受的最大 batch size 與較大模型 preset (tiny/small/base) 的可行性，
並紀錄 forward+backward 單 step 的記憶體峰值與估算吞吐量 (samples/s)。

範例:
  python scripts/capacity_probe.py --npz_path windows_v2_all.npz --presets tiny small base \
      --feature_pack_modes off light full --warmup_trials 1 --measure_trials 2

輸出: JSON 與表格摘要。
"""
import argparse, time, json, math, sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
from datasets.smoke_dataset import build_dataloaders
from models.videoswin import build_videoswin3d_preset

FP_MODE_COMPONENTS = {
    'off':  dict(enable=False, components=None),
    'light':dict(enable=True,  components={'velocity': True, 'accel': True, 'energy': True, 'pairwise': False}),
    'full': dict(enable=True,  components={'velocity': True, 'accel': True, 'energy': True, 'pairwise': True}),
}

def parse_args():
    ap = argparse.ArgumentParser()
    # Default dataset path similar to training script (Slipce_2/windows_v2_all.npz)
    default_npz = Path(__file__).resolve().parents[2] / 'train_data' / 'Slipce_2' / 'windows_v2_all.npz'
    ap.add_argument('--npz_path', default=str(default_npz), help='NPZ dataset path (defaults to common Slipce_2/windows_v2_all.npz)')
    ap.add_argument('--presets', nargs='*', default=['tiny'], help='List of presets to probe')
    ap.add_argument('--preset', type=str, help='Alias for single preset (overrides --presets if provided)')
    ap.add_argument('--initial_batch', type=int, default=4)
    ap.add_argument('--max_batch_cap', type=int, default=256)
    ap.add_argument('--feature_pack_modes', nargs='*', default=['off','light'])
    ap.add_argument('--val_ratio', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--warmup_trials', type=int, default=1)
    ap.add_argument('--measure_trials', type=int, default=2)
    ap.add_argument('--out', type=str, default='capacity_report.json')
    args = ap.parse_args()
    if args.preset:
        args.presets = [args.preset]
    return args

@torch.no_grad()
def _dummy_forward(model, batch):
    logits,_ = model(batch['frames'])
    return logits.mean()

def try_step(model, loader_iter, device, amp, backward=True):
    model.train()
    try:
        batch = next(loader_iter)
    except StopIteration:
        return None, 'data_exhausted'
    x = batch['frames'].to(device)
    y = batch['label'].to(device)
    scaler = torch.cuda.amp.GradScaler() if amp and device.startswith('cuda') else None
    loss_fn = torch.nn.BCEWithLogitsLoss() if model.head.out_features==1 else torch.nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats(device) if device.startswith('cuda') else None
    start=time.time()
    if scaler:
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits,_=model(x)
            if logits.shape[1]==1:
                loss = loss_fn(logits.squeeze(1), y.float())
            else:
                loss = loss_fn(logits, y)
        if backward:
            scaler.scale(loss).backward(); scaler.step(torch.optim.SGD(model.parameters(), lr=1.0)); scaler.update()
    else:
        logits,_=model(x)
        if logits.shape[1]==1:
            loss = loss_fn(logits.squeeze(1), y.float())
        else:
            loss = loss_fn(logits, y)
        if backward:
            loss.backward()
    torch.cuda.synchronize() if device.startswith('cuda') else None
    dur = time.time()-start
    peak = torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0
    return {'loss': float(loss.detach().cpu()), 'time_s': dur, 'peak_mem': peak}, None

def binary_search_batch(build_loader_fn, model_builder, base_batch, max_cap, device, amp):
    lo = base_batch; hi = max_cap; best = None; detail=[]
    while lo <= hi:
        mid = (lo+hi)//2
        try:
            train_loader, _, meta = build_loader_fn(mid)
            model = model_builder(meta).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.01)  # dummy opt just for potential backward
            it = iter(train_loader)
            info, err = try_step(model, it, device, amp, backward=True)
            if err or info is None:
                raise RuntimeError('data issue')
            detail.append({'batch':mid,'ok':True,'peak':info['peak_mem'],'time_s':info['time_s']})
            best = {'batch':mid,'peak':info['peak_mem'],'time_s':info['time_s'],'meta':meta}
            lo = mid + 1
            del model, opt, train_loader
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                detail.append({'batch':mid,'ok':False,'error':'OOM'})
                hi = mid - 1
                torch.cuda.empty_cache()
            else:
                detail.append({'batch':mid,'ok':False,'error':str(e)[:120]})
                hi = mid - 1
    return best, detail

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    npz = Path(args.npz_path)
    if not npz.exists():
        # auto-resolve similar to training script
        search_roots = [
            Path.cwd(),
            Path.cwd()/ 'train_data',
            Path.cwd()/ 'train_data' / 'Slipce_2',
            BASE_DIR.parent / 'train_data',
        ]
        found = []
        for root in search_roots:
            if root.exists():
                cand = list(root.glob(npz.name))
                if cand:
                    found.extend(cand)
        if not found:
            try:
                found = list(Path.cwd().rglob(npz.name))[:1]
            except Exception:
                found = []
        if found:
            npz = found[0].resolve()
            print(f"[npz_resolve] Input path not found, auto-resolved to: {npz}")
        else:
            print(f'[error] npz not found: {npz}'); return
    for preset in args.presets:
        for fp_mode in args.feature_pack_modes:
            if fp_mode not in FP_MODE_COMPONENTS:
                print(f'[warn] skip unknown fp_mode {fp_mode}')
                continue
            fp_cfg = FP_MODE_COMPONENTS[fp_mode]
            def build_loader_fn(batch_size):
                return build_dataloaders(
                    npz_path=str(npz), batch_size_micro=batch_size, val_ratio=args.val_ratio, seed=args.seed,
                    num_workers=args.num_workers, balance_by_class=False, amplify_hard_negative=False, hard_negative_factor=2.0,
                    temporal_jitter=0, feature_grid=(6,6), replicate_channels=3, use_sampler=False,
                    use_feature_pack=fp_cfg['enable'],
                    fp_velocity=(fp_cfg['components'] or {}).get('velocity', True),
                    fp_accel=(fp_cfg['components'] or {}).get('accel', True),
                    fp_energy=(fp_cfg['components'] or {}).get('energy', True),
                    fp_pairwise=(fp_cfg['components'] or {}).get('pairwise', False),
                    fp_joints=15, fp_dims_per_joint=4, fp_pairwise_subset=20)
            def model_builder(meta):
                in_chans = meta.get('C',3)
                single_logit = True
                return build_videoswin3d_preset(preset, in_chans=in_chans, num_classes=1 if single_logit else 2, window_size=(2,7,7), use_checkpoint=False)
            best, detail = binary_search_batch(build_loader_fn, model_builder, args.initial_batch, args.max_batch_cap, device, args.amp)
            if best:
                # estimate throughput (samples/sec) from best time
                samples = best['batch']
                t = best['time_s']
                thr = samples / t if t>0 else None
            else:
                thr = None
            results.append({
                'preset': preset,
                'feature_pack_mode': fp_mode,
                'best': best,
                'throughput_samples_per_s': thr,
                'detail': detail
            })
            print(f"[probe] preset={preset} fp={fp_mode} best_batch={best and best['batch']} peak={(best and best['peak'])} thr={thr}")
    # Summarize recommendation: choose highest throughput under 90% of peak mem among presets
    recommend = None
    max_thr = -1
    for r in results:
        b = r.get('best')
        if not b: continue
        if r['throughput_samples_per_s'] and r['throughput_samples_per_s']>max_thr:
            max_thr = r['throughput_samples_per_s']; recommend = {'preset':r['preset'],'feature_pack_mode':r['feature_pack_mode'],'batch':b['batch'],'peak_mem':b['peak'],'throughput':r['throughput_samples_per_s']}
    # JSON safety helper
    def _safe(o):
        import numpy as _np, torch as _t
        if isinstance(o, (_np.integer,)): return int(o)
        if isinstance(o, (_np.floating,)): return float(o)
        if isinstance(o, (_np.ndarray,)):
            return o.tolist()
        if isinstance(o, _t.Tensor):
            if o.ndim==0:
                return o.item()
            return o.detach().cpu().tolist()
        if isinstance(o, dict):
            return {k:_safe(v) for k,v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_safe(x) for x in o]
        return o
    report = _safe({'device':device,'torch':torch.__version__,'results':results,'recommend':recommend})
    with open(args.out,'w') as f:
        json.dump(report,f,indent=2)
    print('[done] wrote', args.out)
    if recommend:
        print('[suggest] preset={preset} fp_mode={fp} batch={batch} ~throughput={thr:.2f} samples/s'.format(
            preset=recommend['preset'], fp=recommend['feature_pack_mode'], batch=recommend['batch'], thr=recommend['throughput']))

if __name__=='__main__':
    main()
