#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys, time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    import yaml  # type: ignore
except Exception as e:  # provide friendly message if missing
    raise RuntimeError("需要 PyYAML (pip install pyyaml) 以讀取設定檔") from e

# from configs import videoswin_smoke  # (若未轉為 py 模組可忽略)
from datasets.smoke_dataset import build_dataloaders
from models.videoswin import build_videoswin3d_feature
from models.checkpointing import apply_activation_checkpointing
from trainers.trainer import Trainer
from utils.logger import get_logger, append_jsonl
from utils.seed import set_seed
from utils.schedulers import build_scheduler


def load_yaml(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(BASE_DIR / 'configs' / 'videoswin_smoke.yaml'))
    ap.add_argument('--epochs', type=int, default=None, help='Override epochs in config for quick test')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['training']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['device'] = device

    # build data
    tr_loader, va_loader, meta = build_dataloaders(
        npz_path=cfg['data']['npz_path'], batch_size_micro=cfg['data']['batch_size_micro'],
        val_ratio=0.2, seed=cfg['training']['seed'], num_workers=cfg['data']['num_workers'],
        balance_by_class=cfg['data']['balance_by_class'], amplify_hard_negative=cfg['data']['amplify_hard_negative'],
        hard_negative_factor=cfg['data']['hard_negative_factor'], temporal_jitter=cfg['data']['temporal_jitter'],
        feature_grid=tuple(cfg['data']['feature_grid']), replicate_channels=cfg['data']['replicate_channels'])

    # model
    mcfg = cfg['model']
    model = build_videoswin3d_feature(
        in_chans=mcfg['in_chans'], embed_dim=mcfg['embed_dim'], depths=mcfg['depths'], num_heads=mcfg['num_heads'],
        window_size=mcfg['window_size'], mlp_ratio=mcfg['mlp_ratio'], drop_rate=mcfg['drop_rate'],
        attn_drop_rate=mcfg['attn_drop_rate'], drop_path_rate=mcfg['drop_path_rate'], num_classes=mcfg['num_classes'],
        use_checkpoint=mcfg['use_checkpoint'])
    if mcfg['use_checkpoint']:
        apply_activation_checkpointing(model, cfg['checkpointing']['layers_to_ckpt'])
    model.to(device)

    # optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'], betas=tuple(cfg['training']['betas']))
    scheduler = build_scheduler(opt, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['training']['label_smoothing'])
    scaler = GradScaler(enabled=cfg['training']['use_amp'] and device=='cuda')

    # logging dirs
    run_root = BASE_DIR / cfg['experiment']['output_root']
    run_root.mkdir(parents=True, exist_ok=True)
    # auto idx
    idx = 1
    while True:
        run_dir = run_root / f"{idx:02d}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            break
        idx += 1
    if args.tag:
        tagged = run_root / f"{run_dir.name}_{args.tag}"
        if not tagged.exists():
            run_dir.rename(tagged); run_dir = tagged
    logger = get_logger('videoswin_train', str(run_dir))
    logger.info(f"Run dir: {run_dir}")
    (run_dir / 'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')

    # override epochs if provided
    if args.epochs is not None:
        cfg['training']['epochs'] = int(args.epochs)
    trainer = Trainer(model, opt, scheduler, scaler, criterion, cfg, logger)
    # 以 F1 為早停指標
    best = {'val_f1': -1, 'epoch': -1}
    es_patience = cfg['training'].get('early_stop_patience', 0)
    es_counter = 0
    log_jsonl = run_dir / 'train_log.jsonl'

    for epoch in range(1, cfg['training']['epochs']+1):
        tr_metrics = trainer.train_epoch(tr_loader, epoch)
        if epoch % cfg['log']['val_interval'] == 0:
            va_metrics = trainer.validate(va_loader, epoch)
        else:
            va_metrics = {'val_loss': float('nan'), 'val_acc': float('nan'), 'val_precision': float('nan'), 'val_recall': float('nan'), 'val_f1': float('nan'), 'val_auc': float('nan'), 'val_tn': 0, 'val_fp':0, 'val_fn':0, 'val_tp':0}
        rec = {'epoch': epoch, **tr_metrics, **va_metrics, 'lr': opt.param_groups[0]['lr']}
        append_jsonl(str(log_jsonl), rec)
        logger.info(
                f"E{epoch:03d} train_loss={tr_metrics['train_loss']:.4f} acc={tr_metrics['train_acc']:.4f} | "
                f"val_loss={va_metrics['val_loss']:.4f} acc={va_metrics['val_acc']:.4f} f1={va_metrics.get('val_f1', float('nan')):.4f} auc={va_metrics.get('val_auc', float('nan')):.4f} "
                f"cm=[tn:{va_metrics.get('val_tn','-')}, fp:{va_metrics.get('val_fp','-')}, fn:{va_metrics.get('val_fn','-')}, tp:{va_metrics.get('val_tp','-')}]"
        )
        trainer.step_scheduler(va_metrics)
        improved = (not math.isnan(va_metrics['val_f1'])) and va_metrics['val_f1'] > best['val_f1']
        if improved:
            best = {'val_f1': va_metrics['val_f1'], 'epoch': epoch}
            es_counter = 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_f1': va_metrics['val_f1']}, run_dir / 'best.ckpt')
        else:
            if epoch % cfg['log']['val_interval'] == 0 and es_patience > 0:
                es_counter += 1
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_f1': va_metrics['val_f1']}, run_dir / 'last.ckpt')
        if es_patience > 0 and es_counter >= es_patience:
            logger.info(f"[EARLY-STOP] patience={es_patience} reached. Best epoch={best['epoch']}")
            break
    logger.info(f"Best val_f1={best['val_f1']:.4f} @ epoch {best['epoch']}")

if __name__ == '__main__':
    main()
