#!/usr/bin/env python3
from __future__ import annotations
"""Training script for Video Swin 3D feature model (pseudo video smoke dataset).

Features:
- F1 / AUC / Confusion Matrix metrics with early stopping on F1.
- Gradient accumulation (micro-batch) & AMP.
- Checkpoint saving (best + last) with config.
- Optional activation checkpointing (stage >= 2) via model flag.

Usage (example):
    python scripts/train_videoswin.py \
        --npz_path /path/to/data.npz \
        --preset tiny \
        --epochs 30 --batch 8 --accum 2 --lr 3e-4 \
        --out runs/swin_tiny

"""
import argparse, json
from pathlib import Path
import sys, math, time
import platform, datetime, subprocess
import torch
import torch.nn.functional as F
from torch import nn

try:
    from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix  # type: ignore
except Exception:  # fallback minimal implementations
    import numpy as _np
    def f1_score(y_true, y_pred, average='binary'):
        tp = ((y_true==1)&(y_pred==1)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        precision = tp / max(1, tp+fp)
        recall = tp / max(1, tp+fn)
        return 2*precision*recall/max(1e-8, precision+recall)
    def roc_auc_score(y_true, y_prob):
        # simple rank-based AUC (binary)
        order = _np.argsort(-y_prob)
        y = y_true[order]
        pos = (y==1)
        n_pos = pos.sum(); n_neg = len(y)-n_pos
        if n_pos==0 or n_neg==0:
            raise ValueError('AUC undefined')
        cum_pos = _np.cumsum(pos)
        rank_pos_sum = cum_pos[pos].sum()
        return (rank_pos_sum - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    def confusion_matrix(y_true, y_pred):
        tp = ((y_true==1)&(y_pred==1)).sum()
        tn = ((y_true==0)&(y_pred==0)).sum()
        fp = ((y_true==0)&(y_pred==1)).sum()
        fn = ((y_true==1)&(y_pred==0)).sum()
        return _np.array([[tn, fp],[fn, tp]])


# Ensure base path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from models.videoswin import build_videoswin3d_preset, build_videoswin3d_feature, VideoSwin3DConfig
from datasets.smoke_dataset import build_dataloaders


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_path', required=True)
    ap.add_argument('--preset', default='tiny', help='tiny|small|base or custom if --depths provided')
    ap.add_argument('--depths', type=int, nargs='*', help='Override depths list')
    ap.add_argument('--num_heads', type=int, nargs='*', help='Override num_heads list')
    ap.add_argument('--embed_dim', type=int, help='Override embed_dim')
    ap.add_argument('--window_size', type=int, nargs=3, default=(2,7,7))
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=1, help='Gradient accumulation steps')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--warmup_epochs', type=int, default=2)
    ap.add_argument('--drop_path_rate', type=float, help='Override drop path')
    ap.add_argument('--use_checkpoint', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    # Default輸出改為專案根目錄下 result_mod（內部仍會再自動建立 01,02,...）
    default_out = Path(__file__).resolve().parents[1] / 'result_mod'
    ap.add_argument('--out', default=str(default_out))
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--temporal_jitter', type=int, default=0)
    ap.add_argument('--feature_grid', type=int, nargs=2, default=(6,6))
    ap.add_argument('--replicate_channels', type=int, default=3)
    ap.add_argument('--balance_by_class', action='store_true')
    ap.add_argument('--amplify_hard_negative', action='store_true')
    ap.add_argument('--hard_negative_factor', type=float, default=2.0)
    ap.add_argument('--early_patience', type=int, default=6)
    ap.add_argument('--min_delta', type=float, default=1e-4)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--softmax', action='store_true', help='Use 2-logit softmax (default=single-logit BCE)')
    ap.add_argument('--debug', action='store_true', help='Print probability/AUC diagnostics')
    ap.add_argument('--focal_loss', action='store_true', help='Apply focal loss')
    ap.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for focal loss')
    ap.add_argument('--tensorboard', action='store_true', help='Log to TensorBoard (if tensorboard installed)')
    ap.add_argument('--no_sampler', action='store_true', help='Disable weighted sampler; use plain shuffle + optional pos_weight')
    ap.add_argument('--pos_weight', type=float, default=None, help='Pos class weight for BCE / CE (applies to positive class)')
    ap.add_argument('--ema_f1', type=float, default=0.2, help='Exponential moving average factor for F1 tracking (0 to disable)')
    ap.add_argument('--no_early_stop', action='store_true', help='Disable early stopping regardless of patience')
    return ap.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_model(args, num_classes: int):
    overrides = {}
    if args.depths: overrides['depths'] = tuple(args.depths)
    if args.num_heads: overrides['num_heads'] = tuple(args.num_heads)
    if args.embed_dim: overrides['embed_dim'] = args.embed_dim
    if args.drop_path_rate is not None: overrides['drop_path_rate'] = args.drop_path_rate
    if overrides:
        # Build from preset then override
        model = build_videoswin3d_preset(args.preset, num_classes=num_classes, window_size=tuple(args.window_size),
                                         use_checkpoint=args.use_checkpoint, **overrides)
    else:
        model = build_videoswin3d_preset(args.preset, num_classes=num_classes, window_size=tuple(args.window_size),
                                         use_checkpoint=args.use_checkpoint)
    return model


def cosine_lr(step, total_steps, base_lr, min_lr=1e-6, warmup_steps=0):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def train_one_epoch(model, loader, optimizer, scaler, device, accum, global_step, total_steps, base_lr, warmup_steps, single_logit: bool, use_focal: bool, gamma: float, pos_weight: float | None):
    model.train()
    total_loss = 0.0
    last_lr = base_lr
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        frames = batch['frames'].to(device)  # (B,T,C,H,W)
        labels = batch['label'].to(device)
        # Permute to (B,T,C,H,W) already correct from collate
        with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu', enabled=scaler is not None):
            logits, _ = model(frames)
            if single_logit:
                if logits.shape[1] != 1:
                    raise ValueError('Expected single logit output (B,1) in single-logit mode')
                logit1 = logits.squeeze(1)
                pw_tensor = None
                if pos_weight is not None:
                    pw_tensor = torch.tensor(pos_weight, device=logit1.device)
                if use_focal:
                    bce = F.binary_cross_entropy_with_logits(logit1, labels.float(), reduction='none')
                    p = torch.sigmoid(logit1)
                    p_t = p*labels + (1-p)*(1-labels)
                    if pw_tensor is not None:
                        bce = bce * torch.where(labels==1, pw_tensor, torch.tensor(1.0, device=logit1.device))
                    loss_val = (bce * (1 - p_t).pow(gamma)).mean()
                else:
                    if pw_tensor is not None:
                        loss_val = F.binary_cross_entropy_with_logits(logit1, labels.float(), pos_weight=pw_tensor)
                    else:
                        loss_val = F.binary_cross_entropy_with_logits(logit1, labels.float())
                loss = loss_val / accum
            else:
                if use_focal:
                    ce = F.cross_entropy(logits, labels, reduction='none')
                    p_full = torch.softmax(logits, dim=-1)
                    p_t = p_full[torch.arange(logits.size(0), device=logits.device), labels]
                    if pos_weight is not None:
                        wt = torch.where(labels==1, torch.tensor(pos_weight, device=logits.device), torch.tensor(1.0, device=logits.device))
                        ce = ce * wt
                    loss_val = (ce * (1 - p_t).pow(gamma)).mean()
                    loss = loss_val / accum
                else:
                    if pos_weight is not None:
                        cw = torch.ones(logits.size(1), device=logits.device)
                        cw[1] = pos_weight
                        loss = F.cross_entropy(logits, labels, weight=cw) / accum
                    else:
                        loss = F.cross_entropy(logits, labels) / accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        total_loss += loss.item() * accum
        step_this = global_step + i
        lr = cosine_lr(step_this, total_steps, base_lr, warmup_steps=warmup_steps)
        for pg in optimizer.param_groups: pg['lr'] = lr
        last_lr = lr
        if (i + 1) % accum == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / max(1, len(loader)), last_lr


def evaluate(model, loader, device, single_logit: bool, debug: bool):
    model.eval()
    ys=[]; ps=[]; probs=[]
    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            logits, _ = model(frames)
            if single_logit:
                prob = torch.sigmoid(logits.squeeze(1))  # (B,)
                pred = (prob >= 0.5).long()
                ys.append(labels.cpu()); ps.append(pred.cpu()); probs.append(prob.cpu())
            else:
                prob_full = logits.softmax(-1)
                prob = prob_full[:,1]
                pred = prob_full.argmax(-1)
                ys.append(labels.cpu()); ps.append(pred.cpu()); probs.append(prob.cpu())
    import torch as _t
    y = _t.cat(ys); p = _t.cat(ps); pr = _t.cat(probs)
    f1 = f1_score(y.numpy(), p.numpy(), average='binary')
    try:
        auc = roc_auc_score(y.numpy(), pr.numpy())
    except ValueError:
        auc = float('nan')
    cm = confusion_matrix(y.numpy(), p.numpy())
    acc = (p==y).float().mean().item()
    # ---- Optional debug diagnostics ----
    _y_np = y.numpy(); _pr_np = pr.numpy()
    import numpy as _np
    pr_min, pr_max = float(_pr_np.min()), float(_pr_np.max())
    pr_mean, pr_std = float(_pr_np.mean()), float(_pr_np.std())
    pos_scores = _pr_np[_y_np==1]; neg_scores = _pr_np[_y_np==0]
    if pos_scores.size>0 and neg_scores.size>0:
        pos_mean, pos_std = float(pos_scores.mean()), float(pos_scores.std())
        neg_mean, neg_std = float(neg_scores.mean()), float(neg_scores.std())
    else:
        pos_mean=pos_std=neg_mean=neg_std=float('nan')
    try:
        rev_auc = roc_auc_score(_y_np, 1.0 - _pr_np)
    except Exception:
        rev_auc = float('nan')
    if debug:
        module_src = getattr(roc_auc_score, '__module__', 'unknown')
        try:
            order = _np.argsort(_pr_np)
            y_ord = _y_np[order]
            n_pos = y_ord.sum(); n_neg = len(y_ord) - n_pos
            if n_pos>0 and n_neg>0:
                ranks = _np.arange(1, len(y_ord)+1)
                pos_ranks = ranks[y_ord==1]
                manual_auc = (pos_ranks.mean() - (n_pos+1)/2) / n_neg
            else:
                manual_auc = float('nan')
        except Exception:
            manual_auc = float('nan')
        valN = len(_y_np); val_pos = int(_y_np.sum()); val_neg = valN - val_pos
        print(f"[DEBUG] mode={'BCE1' if single_logit else 'SMX2'} module={module_src} valN={valN} pos={val_pos} neg={val_neg} prob_min={pr_min:.4f} prob_max={pr_max:.4f} mean={pr_mean:.4f} std={pr_std:.6f} pos_mean={pos_mean:.4f} pos_std={pos_std:.6f} neg_mean={neg_mean:.4f} neg_std={neg_std:.6f} auc={auc:.4f} rev_auc={rev_auc:.4f} manual_auc={manual_auc:.4f}")
    # ---------------------------------------
    return {'f1':f1,'auc':auc,'cm':cm.tolist(),'acc':acc}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_out = Path(args.out)
    # Auto-increment: if base_out exists and is intended as a container, create numbered subfolder 01,02,...
    # Rule: if base_out is a directory path (whether exists or not) and does not already end with a numeric token, create next index.
    def _next_run_dir(root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        existing = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name)<=3]
        if not existing:
            return root / '01'
        nums = sorted(int(p.name) for p in existing)
        nxt = nums[-1] + 1
        return root / f"{nxt:02d}"
    # If user provided a path whose last component is all digits, respect it (no auto create inside).
    if base_out.name.isdigit():
        out = base_out
    else:
        # If path doesn't exist OR exists but contains previous numeric runs -> allocate next.
        numeric_subdirs = [p for p in base_out.glob('[0-9][0-9]') if p.is_dir()]
        if not base_out.exists() or numeric_subdirs:
            out = _next_run_dir(base_out)
        else:
            # Empty/new directory: use it directly as first run
            out = base_out / '01'
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run_dir] Using output directory: {out}")

    train_loader, val_loader, meta = build_dataloaders(
        npz_path=args.npz_path, batch_size_micro=args.batch, val_ratio=args.val_ratio, seed=args.seed,
        num_workers=args.num_workers, balance_by_class=args.balance_by_class, amplify_hard_negative=args.amplify_hard_negative,
        hard_negative_factor=args.hard_negative_factor, temporal_jitter=args.temporal_jitter, feature_grid=tuple(args.feature_grid),
        replicate_channels=args.replicate_channels, use_sampler=(not args.no_sampler))

    # Auto derive pos_weight if requested (sampler disabled) and counts available
    if args.pos_weight is None and args.no_sampler:
        counts = meta.get('train_class_counts') if isinstance(meta, dict) else None
        if counts is not None and len(counts) >= 2 and counts[1] > 0:
            try:
                args.pos_weight = float(counts[0] / counts[1])
                print(f"[auto] Derived pos_weight={args.pos_weight:.4f} from class counts neg={counts[0]} pos={counts[1]}")
            except Exception:
                pass

    single_logit = (not args.softmax)
    num_classes = 1 if single_logit else 2
    model = build_model(args, num_classes).to(device)

    # Optimizer (AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.startswith('cuda') else None

    best_f1 = -1.0; best_epoch=-1; patience=0
    history = []

    config_dump = {
        'data': {
            'npz_path': args.npz_path,
            'feature_grid': list(args.feature_grid),
            'replicate_channels': args.replicate_channels,
        },
        'model': {
            'preset': args.preset,
            'depths': list(model.stages[0]['blocks'].__len__() for _ in [0]),  # placeholder
        },
        'training': {
            'epochs': args.epochs,
            'batch': args.batch,
            'accum': args.accum,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
        }
    }

    ema_f1 = None
    for epoch in range(1, args.epochs+1):
        start = time.time()
        loss, last_lr = train_one_epoch(model, train_loader, optimizer, scaler, device, args.accum,
                               (epoch-1)*len(train_loader), total_steps, args.lr, warmup_steps, single_logit, args.focal_loss, args.focal_gamma, args.pos_weight)
        metrics = evaluate(model, val_loader, device, single_logit, args.debug)
        if args.ema_f1 > 0:
            if ema_f1 is None:
                ema_f1 = metrics['f1']
            else:
                ema_f1 = args.ema_f1 * metrics['f1'] + (1 - args.ema_f1) * ema_f1
            metrics['f1_ema'] = ema_f1
        history.append({'epoch': epoch, 'loss': loss, 'lr': last_lr, **metrics})
        # TensorBoard logging
        if args.tensorboard:
            if 'writer' not in globals():
                try:
                    from torch.utils.tensorboard import SummaryWriter  # type: ignore
                    globals()['writer'] = SummaryWriter(log_dir=str(out / 'tb'))
                except Exception:
                    globals()['writer'] = None
            w = globals().get('writer', None)
            if w is not None:
                w.add_scalar('train/loss', loss, epoch)
                w.add_scalar('train/lr', last_lr, epoch)
                w.add_scalar('val/f1', metrics['f1'], epoch)
                w.add_scalar('val/auc', metrics['auc'], epoch)
                w.add_scalar('val/acc', metrics['acc'], epoch)
        target_f1 = metrics['f1_ema'] if 'f1_ema' in metrics else metrics['f1']
        improved = target_f1 > best_f1 + args.min_delta
        if improved:
            best_f1 = target_f1; best_epoch=epoch; patience=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics}, out/'best.ckpt')
        else:
            patience+=1
        torch.save({'model':model.state_dict(),'epoch':epoch,'metrics':metrics}, out/'last.ckpt')
        with (out/'history.json').open('w') as f:
            json.dump(history,f,indent=2)
        extra = f" f1_ema={metrics['f1_ema']:.4f}" if 'f1_ema' in metrics else ''
        print(f"Epoch {epoch} loss={loss:.4f} f1={metrics['f1']:.4f}{extra} auc={metrics['auc']:.4f} acc={metrics['acc']:.4f} best_f1={best_f1:.4f} ({best_epoch}) time={time.time()-start:.1f}s")
        if (not args.no_early_stop) and patience >= args.early_patience:
            print('[EARLY STOP]')
            break

    if args.tensorboard and globals().get('writer', None) is not None:
        try:
            globals()['writer'].close()
        except Exception:
            pass
    print('Training complete. Best epoch', best_epoch, 'F1', best_f1)

    # ---------------- Reproducibility Report -----------------
    try:
        git_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=BASE_DIR, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_commit = 'N/A'
    try:
        cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    except Exception:
        cuda_name = 'unknown'
    param_count = sum(p.numel() for p in model.parameters())
    best_rec = None
    last_rec = None
    if history:
        # history list of dicts
        for h in history:
            if h.get('epoch') == best_epoch:
                best_rec = h
        last_rec = history[-1]
    # Helper to make json-safe
    def _safe(o):
        try:
            import numpy as _np
            if isinstance(o, (_np.integer, )): return int(o)
            if isinstance(o, (_np.floating, )): return float(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        if isinstance(o, (list, tuple)):
            return [ _safe(x) for x in o ]
        return o

    repro_json = {
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
        'command': ' '.join(sys.argv),
        'git_commit': git_commit,
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_name': cuda_name,
        'seed': args.seed,
        'data_npz': args.npz_path,
        'data_meta': {
            'train_class_counts': _safe(meta.get('train_class_counts')) if isinstance(meta, dict) else None
        },
        'model': {
            'preset': args.preset,
            'window_size': tuple(args.window_size),
            'depths': _safe(getattr(model, 'depths', None)),
            'num_heads': _safe(getattr(model, 'num_heads', None)),
            'embed_dim': getattr(model, 'embed_dim', None),
            'drop_path_rate': getattr(model, 'drop_path_rate', None),
            'use_checkpoint': args.use_checkpoint,
            'param_count': param_count,
            'single_logit': (not args.softmax)
        },
        'optimization': {
            'epochs': args.epochs,
            'batch_micro': args.batch,
            'accum': args.accum,
            'effective_batch': args.batch * args.accum,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'early_patience': args.early_patience,
            'min_delta': args.min_delta,
            'no_early_stop': args.no_early_stop,
            'pos_weight': args.pos_weight,
            'sampler_used': (not args.no_sampler) and (args.balance_by_class or args.amplify_hard_negative),
            'balance_by_class': args.balance_by_class,
            'amplify_hard_negative': args.amplify_hard_negative,
            'hard_negative_factor': args.hard_negative_factor,
            'temporal_jitter': args.temporal_jitter,
            'focal_loss': args.focal_loss,
            'focal_gamma': args.focal_gamma,
            'ema_f1_alpha': args.ema_f1
        },
        'metrics': {
            'best_epoch': best_epoch,
            'best_f1': best_f1,
            'best_record': best_rec,
            'last_record': last_rec
        },
        'artifacts': {
            'history_json': str((out/'history.json').resolve()),
            'best_ckpt': str((out/'best.ckpt').resolve()),
            'last_ckpt': str((out/'last.ckpt').resolve())
        },
        'notes': 'Weights (.ckpt/.pt/.onnx) may be excluded from backup; this report plus seed & code commit allow retraining. Minor nondeterminism can arise from CUDA kernels unless fully deterministic settings are enforced.'
    }
    with (out/'reproduce.json').open('w') as f:
        json.dump(repro_json, f, indent=2)
    # Human-readable summary
    lines = []
    lines.append('# Reproducibility Report')
    lines.append(f"Run dir: {out}")
    lines.append(f"Timestamp (UTC): {repro_json['timestamp']}")
    lines.append(f"Command: {repro_json['command']}")
    lines.append(f"Git commit: {git_commit}")
    lines.append(f"Python: {repro_json['python']} Torch: {repro_json['torch']} Device: {cuda_name}")
    lines.append('--- Model ---')
    for k,v in repro_json['model'].items():
        lines.append(f"{k}: {v}")
    lines.append('--- Optimization ---')
    for k,v in repro_json['optimization'].items():
        lines.append(f"{k}: {v}")
    if best_rec:
        lines.append('--- Best Epoch Metrics ---')
        for k,v in best_rec.items():
            lines.append(f"{k}: {v}")
    if last_rec:
        lines.append('--- Last Epoch Metrics ---')
        for k,v in last_rec.items():
            lines.append(f"{k}: {v}")
    lines.append('--- Reproduce Command (suggested) ---')
    reproduce_cmd = [sys.executable, __file__, f"--npz_path {args.npz_path}", f"--preset {args.preset}", f"--epochs {args.epochs}", f"--batch {args.batch}", f"--accum {args.accum}", f"--lr {args.lr}", f"--warmup_epochs {args.warmup_epochs}", f"--seed {args.seed}"]
    if args.no_sampler: reproduce_cmd.append('--no_sampler')
    if args.softmax: reproduce_cmd.append('--softmax')
    if args.use_checkpoint: reproduce_cmd.append('--use_checkpoint')
    if args.balance_by_class: reproduce_cmd.append('--balance_by_class')
    if args.amplify_hard_negative: reproduce_cmd.append('--amplify_hard_negative')
    if args.focal_loss: reproduce_cmd.append(f"--focal_loss --focal_gamma {args.focal_gamma}")
    if args.pos_weight is not None: reproduce_cmd.append(f"--pos_weight {args.pos_weight}")
    if args.ema_f1 != 0: reproduce_cmd.append(f"--ema_f1 {args.ema_f1}")
    if args.temporal_jitter: reproduce_cmd.append(f"--temporal_jitter {args.temporal_jitter}")
    reproduce_cmd.append(f"--val_ratio {args.val_ratio}")
    lines.append(' '.join(reproduce_cmd))
    lines.append('--- Notes ---')
    lines.append(repro_json['notes'])
    with (out/'reproduce.txt').open('w') as f:
        f.write('\n'.join(lines)+'\n')
    print('[report] Wrote reproduce.json and reproduce.txt')

if __name__=='__main__':
    main()
