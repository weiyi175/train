from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List


def get_next_run_dir(base: str) -> Path:
    base_p = Path(base)
    base_p.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        d = base_p / f"{idx:02d}"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            return d
        idx += 1


def save_json(obj: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _load_logs(path: Path) -> List[Dict[str, Any]]:
    logs = []
    if not path.exists():
        return logs
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except Exception:
                pass
        # ensure logs are ordered by epoch (some tools may append out-of-order)
        try:
            logs = sorted(logs, key=lambda x: int(x.get('epoch', 0)))
        except Exception:
            pass
        return logs


def _slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / max(1, len(values) - 1)


def generate_run_report(run_dir: str):
    run = Path(run_dir)
    cfg = json.load(open(run / 'config.json', 'r', encoding='utf-8'))
    logs = _load_logs(run / 'train_log.jsonl')
    if not logs:
        return
    # determine actual last epoch from logs and from checkpoint if present
    # monitor primarily by val_loss (lower better), secondary val_acc (higher better)
    best = min(logs, key=lambda x: (x.get('val_loss', float('inf')), -x.get('val_acc', float('-inf'))))
    # prefer checkpoint epoch info if available
    last_epoch_from_logs = int(logs[-1].get('epoch', 0))
    epochs = last_epoch_from_logs
    last = logs[-1]
    # try to read last.ckpt for authoritative epoch if present
    ckpt_epoch = None
    for ck in ('last.ckpt', 'best.ckpt'):
        p = run / ck
        if p.exists():
            try:
                obj = json.load(open(p, 'r', encoding='utf-8'))
                # some ckpt saved as json-like dict
                if 'epoch' in obj:
                    ckpt_epoch = int(obj['epoch'])
                    break
            except Exception:
                # try torch load if available
                try:
                    import torch
                    obj = torch.load(str(p), map_location='cpu')
                    if isinstance(obj, dict) and 'epoch' in obj:
                        ckpt_epoch = int(obj['epoch'])
                        break
                except Exception:
                    pass
    if ckpt_epoch is not None:
        # prefer ckpt epoch as authoritative last epoch
        epochs = max(epochs, ckpt_epoch)
    best_epoch = int(best.get('epoch', 0))
    # top-K best epochs by val_loss (asc) then val_acc (desc)
    K_top = 4
    sorted_by_val = sorted(logs, key=lambda x: (x.get('val_loss', float('inf')), -x.get('val_acc', float('-inf'))))
    top_k = []
    for entry in sorted_by_val[:K_top]:
        top_k.append({
            'epoch': int(entry.get('epoch', -1)),
            'train_loss': float(entry.get('train_loss', float('nan'))),
            'train_acc': float(entry.get('train_acc', float('nan'))),
            'val_loss': float(entry.get('val_loss', float('nan'))),
            'val_acc': float(entry.get('val_acc', float('nan'))),
        })
    best_ratio = best_epoch / max(1, epochs)
    gap_best = best['train_loss'] - best['val_loss']
    gap_last = last['train_loss'] - last['val_loss']
    K = min(10, len(logs))
    seg = logs[-K:]
    train_loss_slope = _slope([x['train_loss'] for x in seg])
    train_acc_slope = _slope([x['train_acc'] for x in seg])
    val_loss_slope = _slope([x['val_loss'] for x in seg])
    val_acc_slope = _slope([x['val_acc'] for x in seg])
    early_best = best_ratio < 0.4
    loss_rebound = val_loss_slope > 0.0
    gap_large = (gap_last - gap_best) > 0.1
    acc_drop = (best['val_acc'] - last['val_acc']) > 0.02
    score = int(early_best) + int(loss_rebound) + int(gap_large) + int(acc_drop)
    overfit = score >= 2
    report_json = {
        'model': cfg.get('model', 'tcn_attention'),
        'split': cfg.get('split', 'short'),
        'device': cfg.get('device', 'cpu'),
        'params': cfg.get('params', {}),
        'data': cfg.get('data', {}),
            'core_metrics': {
            'best': best,
            'last': last,
            'generalization_gap': {'at_best': gap_best, 'at_last': gap_last},
            'top_k': top_k,
        },
        'monitor': {'primary': 'val_loss', 'secondary': 'val_acc'},
        'trends': {
            'train_loss_slope': train_loss_slope,
            'train_acc_slope': train_acc_slope,
            'val_loss_slope': val_loss_slope,
            'val_acc_slope': val_acc_slope,
        },
        'overfitting': {
            'is_overfit': overfit,
            'score': score,
            'signals': {
                'early_best': early_best,
                'loss_rebound': loss_rebound,
                'gap_large': gap_large,
                'acc_drop': acc_drop,
                'best_epoch_ratio': round(best_ratio, 2),
            },
        },
        'hparams': cfg,
    }
    save_json(report_json, str(run / 'report.json'))
    md = []
    md.append('# 訓練報告')
    md.append(f"- 模型: tcn_attention  | 分割: {cfg.get('split')}  | 裝置: {cfg.get('device')}  | 參數量: {cfg.get('num_params', 'NA')}")
    d = cfg.get('data', {})
    md.append(f"- 資料: N={d.get('N','?')} T={d.get('T','?')} F={d.get('F','?')}  | 批次: {cfg.get('batch_size')}  | epoch: {epochs}")
    md.append(f"- 資料: N={d.get('N','?')} T={d.get('T','?')} F={d.get('F','?')}  | 批次: {cfg.get('batch_size')}  | epoch: {epochs}")
    # annotate if config declared epochs differs from actual detected epochs
    declared_epochs = cfg.get('epochs')
    if declared_epochs is not None and int(declared_epochs) != int(epochs):
        md.append(f"- NOTE: config.epochs={declared_epochs} but detected {epochs} epochs in logs/checkpoints")
    md.append('\n## 核心指標')
    md.append(f"- 最佳 (epoch {best_epoch}): train_loss={best['train_loss']:.4f}, train_acc={best['train_acc']:.4f}, val_loss={best['val_loss']:.4f}, val_acc={best['val_acc']:.4f}")
    md.append(f"- 最終 (epoch {last.get('epoch', '?')}): train_loss={last['train_loss']:.4f}, train_acc={last['train_acc']:.4f}, val_loss={last['val_loss']:.4f}, val_acc={last['val_acc']:.4f}")
    md.append(f"- 一般化落差: at_best={gap_best:.4f}, at_last={gap_last:.4f}")
    md.append('\n## 趨勢 (最後 10 個 epoch 粗略斜率)')
    md.append(f"- train_loss_slope: {train_loss_slope:.4f}")
    md.append(f"- train_acc_slope: {train_acc_slope:.4f}")
    md.append(f"- val_loss_slope: {val_loss_slope:.4f}")
    md.append(f"- val_acc_slope: {val_acc_slope:.4f}")
    # learning rate suggestion heuristic
    lr_suggestion = '維持'
    lr_reason = ''
    if loss_rebound or val_loss_slope > 0.0:
        lr_suggestion = '調低'
        lr_reason = 'val_loss 上升或震盪，可能步長過大或不穩定，建議調低 learning rate。'
    elif abs(val_loss_slope) < 1e-4:
        lr_suggestion = '調低'
        lr_reason = 'val_loss 近似平穩（plateau），建議調低 learning rate 以精細化收斂。'
    else:
        lr_suggestion = '維持'
        lr_reason = 'val_loss 持續下降，暫時維持目前 learning rate。'
    md.append('\n## 學習率建議')
    md.append(f"- 建議: {lr_suggestion}  | 當前 lr: {cfg.get('lr')} ")
    md.append(f"- 理由: {lr_reason}")

    md.append('\n## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)')
    if top_k:
        for i, e in enumerate(top_k, 1):
            md.append(f"{i}. epoch {e['epoch']}: train_loss={e['train_loss']:.4f}, train_acc={e['train_acc']:.4f}, val_loss={e['val_loss']:.4f}, val_acc={e['val_acc']:.4f}")
    else:
        md.append('- 無可用的紀錄')
    md.append('\n## 過擬合分析')
    md.append(f"- 判定: {'是' if overfit else '否'} (score={score})")
    md.append(f"- 訊號: early_best={early_best}, loss_rebound={loss_rebound}, gap_large={gap_large}, acc_drop={acc_drop}")
    md.append(f"- 附註: best_epoch_ratio={best_ratio:.2f}")
    md.append('\n## 設定摘要')
    for k in ['lr','weight_decay','seed','use_norm','balance_by_class','amplify_hard_negative','hard_negative_factor','temporal_jitter_frames','val_ratio','num_workers']:
        md.append(f"- {k}: {cfg.get(k)}")
    (run / 'report.md').write_text('\n'.join(md), encoding='utf-8')
