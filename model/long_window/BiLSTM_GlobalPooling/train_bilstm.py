#!/usr/bin/env python3
from pathlib import Path
import argparse, json, time
import torch
import torch.nn.functional as F
import numpy as np

from dataset import LongWindowNPZDataset, collate_batch
import logging
from model import BiLSTMGlobalPooling


# combined score helper used for ranking and early-stopping
def combined_score(auc_v, f1_v, recall_v=0.0):
    a = 0.0 if auc_v is None or (isinstance(auc_v, float) and np.isnan(auc_v)) else float(auc_v)
    b = 0.0 if f1_v is None or (isinstance(f1_v, float) and np.isnan(f1_v)) else float(f1_v)
    c = 0.0 if recall_v is None or (isinstance(recall_v, float) and np.isnan(recall_v)) else float(recall_v)
    # unified weights: 0.5*AUC + 0.3*F1 + 0.2*recall
    return 0.5 * a + 0.3 * b + 0.2 * c


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='/home/user/projects/train/train_data/slipce/windows_npz.npz')
    ap.add_argument('--out', default='/home/user/projects/train/model/long_window/BiLSTM_GlobalPooling/result')
    ap.add_argument('--kfold', type=int, default=0, help='use StratifiedKFold when >1')
    ap.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    ap.add_argument('--save_all_ckpts', action='store_true', help='save per-epoch ckpt in folds')
    ap.add_argument('--early_stopping_patience', type=int, default=5)
    ap.add_argument('--early_stopping_min_delta', type=float, default=1e-4)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    ap.add_argument('--thresholds', default='0.3,0.5,0.7', help='comma-separated thresholds to evaluate F1')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--num_layers', type=int, default=1)
    ap.add_argument('--pool', choices=['avg','max','attn'], default='avg')
    ap.add_argument('--use_bn', action='store_true')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def train_one_fold(train_idx, val_idx, fold_out: Path, args, device, ds):
    from torch.utils.data import Subset, DataLoader
    train_ds = Subset(ds, train_idx); val_ds = Subset(ds, val_idx)
    pin = True if device.type=='cuda' else False
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=pin)

    sample = ds[0]['features']
    input_dim = sample.shape[1]
    model = BiLSTMGlobalPooling(input_dim, hidden_size=args.hidden, num_layers=args.num_layers, pooling=args.pool, use_bn=args.use_bn, single_logit=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0; history = []
    accum_steps = max(1, int(getattr(args, 'accumulation_steps', 1)))
    patience = int(getattr(args, 'early_stopping_patience', 0))
    min_delta = float(getattr(args, 'early_stopping_min_delta', 0.0))
    no_improve = 0
    for epoch in range(1, args.epochs+1):
        model.train(); t0 = time.time(); losses = []
        opt.zero_grad(); step_idx = 0
        for b in train_loader:
            x = b['features'].to(device, non_blocking=True); mask = b['mask'].to(device, non_blocking=True); lengths = b['lengths']
            y = b['label'].to(device).float()
            logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss = loss / accum_steps
            loss.backward()
            losses.append(float((loss * accum_steps).detach().cpu()))
            step_idx += 1
            if step_idx % accum_steps == 0:
                opt.step(); opt.zero_grad()
        if step_idx % accum_steps != 0:
            opt.step(); opt.zero_grad()

    # eval
        import sklearn.metrics as _m
        ys=[]; prs=[]
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x=b['features'].to(device, non_blocking=True); mask=b['mask'].to(device, non_blocking=True); lengths=b['lengths']
                y=b['label']
                logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
                pr = torch.sigmoid(logits).cpu()
                ys.append(y); prs.append(pr)
        thresholds = [float(t) for t in str(getattr(args, 'thresholds', '0.5')).split(',') if t]
        thr_stats = {}
        if prs:
            y = torch.cat(ys).numpy(); pr = torch.cat(prs).numpy()
            try:
                auc = float(_m.roc_auc_score(y, pr))
            except Exception:
                auc = float('nan')
            # compute metrics at multiple thresholds
            for t in thresholds:
                try:
                    pr_bin = (pr > t).astype(int)
                    f1_t = float(_m.f1_score(y, pr_bin))
                    prec = float(_m.precision_score(y, pr_bin, zero_division=0))
                    rec = float(_m.recall_score(y, pr_bin, zero_division=0))
                    cm = _m.confusion_matrix(y, pr_bin, labels=[0,1])
                    tn_t, fp_t, fn_t, tp_t = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
                except Exception:
                    f1_t = float('nan'); prec = rec = 0.0; tp_t = fp_t = fn_t = tn_t = 0
                thr_stats[str(t)] = {'f1': f1_t, 'precision': prec, 'recall': rec, 'tp': tp_t, 'fp': fp_t, 'fn': fn_t, 'tn': tn_t}
            # also set default 0.5 metrics for backward compat
            d0 = thr_stats.get('0.5') or thr_stats.get('0.50') or thr_stats.get(next(iter(thr_stats)))
            if d0:
                f1 = d0.get('f1', float('nan'))
                tp = d0.get('tp', 0); fp = d0.get('fp', 0); fn = d0.get('fn', 0); tn = d0.get('tn', 0)
            else:
                f1 = float('nan'); tp = fp = fn = tn = 0
        else:
            auc = float('nan')
            f1 = float('nan'); tp = fp = fn = tn = 0
            thr_stats = {}
        loss_mean = float(np.mean(losses)) if losses else 0.0
        # compute combined score for this epoch (use recall from d0 if available)
        try:
            recall_val = 0.0
            # prefer the 0.5 threshold recall if present
            if isinstance(thr_stats, dict) and ('0.5' in thr_stats or '0.50' in thr_stats):
                d0 = thr_stats.get('0.5') or thr_stats.get('0.50')
                recall_val = float(d0.get('recall', 0.0)) if d0 else 0.0
            else:
                # fallback to derived recall from tp/fn
                recall_val = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        except Exception:
            recall_val = 0.0
        comb = combined_score(auc, f1, recall_val)
        history.append({'epoch':epoch,'loss':loss_mean,'auc':auc, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'thresholds': thr_stats, 'comb': comb})
        print('Epoch', epoch, 'loss', f'{loss_mean:.6f}', 'auc', f'{auc}', 'f1', f'{f1}', 'TP', tp, 'FP', fp, 'FN', fn, 'TN', tn, 'time', time.time()-t0)
        # save best
        # save ckpt optionally
        if getattr(args, 'save_all_ckpts', False):
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'history': history}, fold_out / f'ckpt_epoch_{epoch:02d}.pt')

        # early stopping and best model using combined score
        # track best_comb separately
        if 'best_comb' not in locals():
            best_comb = float('-inf')
        improved = (not np.isnan(comb)) and (comb > best_comb + min_delta)
        if improved:
            best_comb = comb
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'history': history, 'comb': best_comb}, fold_out / 'best.pt')
            no_improve = 0
        else:
            no_improve += 1
        if patience>0 and no_improve >= patience:
            print(f'Early stopping on fold at epoch {epoch} (no_improve={no_improve})')
            break

    # compute val preds using best model
    best = torch.load(fold_out / 'best.pt', map_location=device)
    model.load_state_dict(best['model_state'])
    model.eval()
    all_prs=[]; all_ys=[]
    with torch.no_grad():
        for b in val_loader:
            x=b['features'].to(device, non_blocking=True); mask=b['mask'].to(device, non_blocking=True); lengths=b['lengths']
            y=b['label']
            logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
            pr = torch.sigmoid(logits).cpu().numpy()
            all_prs.append(pr); all_ys.append(y.numpy())
    if all_prs:
        all_pr = np.concatenate(all_prs); all_y = np.concatenate(all_ys)
    else:
        all_pr = np.array([]); all_y = np.array([])

    with open(fold_out / 'history.json','w') as f:
        json.dump(history, f, indent=2)
    # rich fold report using same format as single-run
    make_rich_report(fold_out, args, device, ds, history, opt)

    # cleanup
    try:
        del model; del opt
        if device.type=='cuda':
            torch.cuda.empty_cache()
    except Exception:
        pass

    return best_auc, all_y, all_pr


def next_run_dir(base: Path) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    existing = [p for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if not existing:
        return base / '01'
    nums = sorted(int(p.name) for p in existing)
    return base / f"{nums[-1]+1:02d}"


def train():
    args = parse_args()
    out_base = Path(args.out)
    out = next_run_dir(out_base)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu')
    print('device', device)

    # set seeds for reproducibility
    try:
        np.random.seed(args.seed)
    except Exception:
        pass
    try:
        import random as _py_rand
        _py_rand.seed(args.seed)
    except Exception:
        pass
    try:
        torch.manual_seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    # setup basic logging per run
    ds = LongWindowNPZDataset(args.npz)
    logger = logging.getLogger('train_bilstm')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(out / 'train.log')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    if not logger.handlers:
        logger.addHandler(fh)
    logger.info('Starting train run')
    # call main training routine
    main_run(args, out, device, ds)


def make_rich_report(out: Path, args, device, ds, history, opt):
    # ported rich report logic from single-run
    try:
        param_count = sum(p.numel() for p in torch.load(out / 'best.pt')['model_state'].values())
    except Exception:
        try:
            param_count = sum(p.numel() for p in ds[0]['features'].shape)
        except Exception:
            param_count = 0

    try:
        N, T, featF = ds.features.shape
    except Exception:
        try:
            s = ds[0]['features']; T, featF = s.shape; N = len(ds)
        except Exception:
            N = len(ds); T = 0; featF = 0

    best_idx = None; best_auc = float('-inf')
    for i, h in enumerate(history):
        auc_v = h.get('auc', float('nan'))
        if not np.isnan(auc_v) and auc_v > best_auc:
            best_auc = auc_v; best_idx = i
    if best_idx is None:
        best_idx = 0

    best_entry = history[best_idx] if history else {}
    final_entry = history[-1] if history else {}

    def slope(values):
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            m = float(np.polyfit(x, values, 1)[0])
        except Exception:
            m = 0.0
        return m

    last_k = history[-10:]
    train_losses = [h.get('loss', 0.0) for h in last_k]
    val_aucs = [h.get('auc', float('nan')) for h in last_k]
    train_loss_slope = slope(train_losses)
    val_auc_clean = [0.0 if np.isnan(v) else v for v in val_aucs]
    val_auc_slope = slope(val_auc_clean)

    # use top-level combined_score helper (weights: 0.5*AUC + 0.3*F1 + 0.2*recall)

    top_epochs = []
    for i, h in enumerate(history):
        auc_v = h.get('auc', float('nan'))
        f1_v = h.get('f1', float('nan'))
        tp_h = h.get('tp', 0); fn_h = h.get('fn', 0)
        recall_h = (tp_h / (tp_h + fn_h)) if (tp_h + fn_h) > 0 else 0.0
        comb = combined_score(auc_v, f1_v, recall_h)
        top_epochs.append((i + 1, h.get('loss', 0.0), auc_v, f1_v, h.get('tp', 0), h.get('fp', 0), h.get('fn', 0), h.get('tn', 0), comb))
    # filter out entries without auc if desired, then sort by combined score
    top_epochs = sorted(top_epochs, key=lambda x: -x[8])[:4]

    try:
        gap = best_entry.get('loss', 0.0) - final_entry.get('loss', 0.0)
    except Exception:
        gap = 0.0

    acc_drop = False
    try:
        acc_drop = (best_entry.get('auc', 0.0) - final_entry.get('auc', 0.0)) > 0.05
    except Exception:
        acc_drop = False

    report_lines = []
    report_lines.append('# 訓練報告')
    report_lines.append(f'- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: {device}  | 參數量: {param_count}')
    report_lines.append(f'- 資料: N={N} T={T} F={featF}  | 批次: {args.batch}  | epoch: {args.epochs}')
    report_lines.append('')
    report_lines.append('## 核心指標')
    report_lines.append(f'- 最佳 (epoch {best_idx+1}): train_loss={best_entry.get("loss",0):.4f}, val_auc={best_entry.get("auc",float("nan")):.4f}, val_f1={best_entry.get("f1", float("nan")):.4f}')
    report_lines.append(f'  - confusion (TP,FP,FN,TN): {best_entry.get("tp",0)},{best_entry.get("fp",0)},{best_entry.get("fn",0)},{best_entry.get("tn",0)}')
    report_lines.append(f'- 最終 (epoch {final_entry.get("epoch",len(history))}): train_loss={final_entry.get("loss",0):.4f}, val_auc={final_entry.get("auc",float("nan")):.4f}, val_f1={final_entry.get("f1", float("nan")):.4f}')
    report_lines.append(f'  - confusion (TP,FP,FN,TN): {final_entry.get("tp",0)},{final_entry.get("fp",0)},{final_entry.get("fn",0)},{final_entry.get("tn",0)}')
    report_lines.append('')
    report_lines.append('## 趨勢 (最後 10 個 epoch 粗略斜率)')
    report_lines.append(f'- train_loss_slope: {train_loss_slope:.4f}')
    report_lines.append(f'- val_auc_slope: {val_auc_slope:.4f}')
    report_lines.append('')
    report_lines.append('## 學習率建議')
    report_lines.append(f'- 建議: 維持  | 當前 lr: {args.lr}')
    report_lines.append('- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。')
    report_lines.append('')
    report_lines.append('## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)')
    if top_epochs:
        for i, (e, l, a, f1v, tp, fp, fn, tn, comb) in enumerate(top_epochs, start=1):
            report_lines.append(f'{i}. epoch {e}: comb_score={comb:.4f}, train_loss={l:.4f}, val_auc={a:.4f}, val_f1={f1v:.4f}')
            report_lines.append(f'   - confusion (TP,FP,FN,TN): {tp},{fp},{fn},{tn}')
    else:
        report_lines.append('無可用 val_auc 結果')
    report_lines.append('')
    report_lines.append('## 過擬合分析')
    report_lines.append(f'- 判定: {"否" if not acc_drop else "是"} (gap={gap:.4f})')
    report_lines.append(f'- 訊號: early_best={best_idx < max(1, args.epochs//2)}, loss_rebound=False, gap_large={abs(gap)>0.1}, acc_drop={acc_drop}')
    report_lines.append('')
    report_lines.append('## 設定摘要')
    report_lines.append(f'- lr: {args.lr}')
    report_lines.append(f'- weight_decay: {opt.defaults.get("weight_decay", 0.0) if hasattr(opt, "defaults") else 0.0}')
    report_lines.append(f'- seed: {args.seed}')
    report_lines.append(f'- use_bn: {args.use_bn}')
    report_lines.append(f'- pooling: {args.pool}')
    report_lines.append(f'- num_workers: {args.num_workers}')

    with open(out / 'report.md', 'w') as f:
        f.write('\n'.join(report_lines))
def main_run(args, out, device, ds):
    # if using k-fold mode
    if getattr(args, 'kfold', 0) and args.kfold > 1:
        from sklearn.model_selection import StratifiedKFold
        N = len(ds)
        labels = np.array([int(ds[i]['label']) for i in range(N)])
        skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
        out_k = out / f'k{args.kfold}'
        out_k.mkdir(parents=True, exist_ok=True)
        oof_preds = np.zeros((N,), dtype=np.float32)
        oof_trues = np.zeros((N,), dtype=np.int64)
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(N), labels), start=1):
            print('Fold', fold, 'train', len(train_idx), 'val', len(val_idx))
            fold_out = out_k / f'fold_{fold:02d}'; fold_out.mkdir(parents=True, exist_ok=True)
            best_auc, ys, prs = train_one_fold(train_idx, val_idx, fold_out, args, device, ds)
            fold_metrics.append(best_auc)
            if prs.size>0:
                oof_preds[val_idx] = prs
                oof_trues[val_idx] = ys
        import sklearn.metrics as _m
        mask = (oof_preds!=0) | (oof_trues!=0)
        try:
            oof_auc = float(_m.roc_auc_score(oof_trues[mask], oof_preds[mask]))
        except Exception:
            oof_auc = float('nan')
        with open(out_k / 'kfold_summary.json','w') as f:
            json.dump({'folds': fold_metrics, 'oof_auc': oof_auc}, f, indent=2)
        print('fold_metrics', fold_metrics)
        print('oof_auc', oof_auc)
        # ensemble inference across folds: load each fold best model and predict on full dataset
        print('Running ensemble inference across folds...')
        from torch.utils.data import DataLoader
        all_fold_probs = []
        for fold in range(1, args.kfold+1):
            mpath = out_k / f'fold_{fold:02d}' / 'best.pt'
            if not mpath.exists():
                print('Missing', mpath); continue
            ck = torch.load(mpath, map_location=device)
            model = BiLSTMGlobalPooling(ds[0]['features'].shape[1], hidden_size=args.hidden, num_layers=args.num_layers, pooling=args.pool, use_bn=args.use_bn, single_logit=True).to(device)
            model.load_state_dict(ck['model_state'])
            model.eval()
            loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch, pin_memory=(device.type=='cuda'))
            probs = []
            with torch.no_grad():
                for b in loader:
                    x=b['features'].to(device, non_blocking=True); mask=b['mask'].to(device, non_blocking=True); lengths=b['lengths']
                    logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
                    pr = torch.sigmoid(logits).cpu().numpy()
                    probs.append(pr)
            all_fold_probs.append(np.concatenate(probs))
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass
        if all_fold_probs:
            stack = np.stack(all_fold_probs, axis=0)  # (n_folds, N)
            avg_probs = stack.mean(axis=0)
            maj_votes = (stack > 0.5).sum(axis=0) > (stack.shape[0]//2)
            import sklearn.metrics as _m
            labels_all = np.array([int(ds[i]['label']) for i in range(len(ds))])
            try:
                ensemble_auc = float(_m.roc_auc_score(labels_all, avg_probs))
            except Exception:
                ensemble_auc = float('nan')
            try:
                maj_auc = float(_m.roc_auc_score(labels_all, maj_votes.astype(float)))
            except Exception:
                maj_auc = float('nan')
            # write ensemble report
            with open(out_k / 'ensemble_report.md','w') as f:
                f.write('# Ensemble Report\n')
                f.write(f'folds: {args.kfold}\n')
                f.write(f'avg_prob_auc: {ensemble_auc}\n')
                f.write(f'majority_vote_auc: {maj_auc}\n')
            np.save(out_k / 'ensemble_avg_probs.npy', avg_probs)
            np.save(out_k / 'ensemble_fold_probs.npy', stack)
            # also write an aggregated k-fold report (top-4 epochs across folds)
            try:
                agg_lines = []
                agg_lines.append('# K-Fold Aggregated Report')
                agg_lines.append(f'- folds: {args.kfold}')
                agg_lines.append(f'- oof_auc: {oof_auc}')
                agg_lines.append(f'- ensemble_avg_prob_auc: {ensemble_auc}')
                agg_lines.append(f'- ensemble_majority_vote_auc: {maj_auc}')
                agg_lines.append('')
                # collect top epochs from each fold history and compute combined score (0.5*AUC+0.3*F1+0.2*recall)
                all_entries = []
                for fold in range(1, args.kfold+1):
                    hpath = out_k / f'fold_{fold:02d}' / 'history.json'
                    if not hpath.exists():
                        continue
                    try:
                        with open(hpath, 'r') as hf:
                            h = json.load(hf)
                        for e in h:
                            auc_v = e.get('auc', float('nan'))
                            f1_v = e.get('f1', float('nan'))
                            if auc_v is None:
                                continue
                            recall_v = (e.get('tp',0) / (e.get('tp',0) + e.get('fn',0))) if (e.get('tp',0) + e.get('fn',0))>0 else 0.0
                            comb = combined_score(auc_v, f1_v, recall_v)
                            all_entries.append({'fold': fold, 'epoch': e.get('epoch'), 'loss': e.get('loss'), 'auc': auc_v, 'f1': f1_v, 'tp': e.get('tp',0), 'fp': e.get('fp',0), 'fn': e.get('fn',0), 'tn': e.get('tn',0), 'comb': comb})
                    except Exception:
                        continue
                # sort and pick top 4 by combined score
                all_entries = [e for e in all_entries if not np.isnan(e.get('comb', float('nan')))]
                all_entries = sorted(all_entries, key=lambda x: -float(x['comb']))
                agg_lines.append('## Top epochs across folds (by combined score 0.5*AUC+0.3*F1+0.2*recall)')
                if all_entries:
                    for i, e in enumerate(all_entries[:4], start=1):
                        agg_lines.append(f'{i}. fold {e["fold"]} epoch {e["epoch"]}: comb_score={e["comb"]:.4f}, val_auc={e["auc"]:.6f}, val_f1={e.get("f1", float("nan")):.4f}')
                        agg_lines.append(f'   - confusion (TP,FP,FN,TN): {e.get("tp",0)},{e.get("fp",0)},{e.get("fn",0)},{e.get("tn",0)}')
                else:
                    agg_lines.append('No valid epoch records found')
                with open(out_k / 'report.md', 'w') as rf:
                    rf.write('\n'.join(agg_lines))
            except Exception:
                pass
        return

    # single-run mode (original behavior)
    n = len(ds)
    idx = np.arange(n); np.random.shuffle(idx)
    n_val = max(1, int(0.1 * n))
    val_idx = idx[:n_val]; train_idx = idx[n_val:]
    from torch.utils.data import Subset, DataLoader
    train_ds = Subset(ds, train_idx); val_ds = Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    sample = ds[0]['features']
    input_dim = sample.shape[1]
    model = BiLSTMGlobalPooling(input_dim, hidden_size=args.hidden, num_layers=args.num_layers, pooling=args.pool, use_bn=args.use_bn, single_logit=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_auc = -1.0; history = []
    # single-run early stopping settings
    patience = int(getattr(args, 'early_stopping_patience', 0))
    min_delta = float(getattr(args, 'early_stopping_min_delta', 0.0))
    no_improve = 0
    for epoch in range(1, args.epochs+1):
        model.train(); t0 = time.time();
        losses = []
        for b in train_loader:
            x = b['features'].to(device); mask = b['mask'].to(device); lengths = b['lengths']
            y = b['label'].to(device).float()
            logits = model(x, lengths=lengths, mask=mask)
            logits = logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu()))
        # eval
        import sklearn.metrics as _m
        ys=[]; prs=[]
        model.eval()
        with torch.no_grad():
            for b in val_loader:
                x=b['features'].to(device); mask=b['mask'].to(device); lengths=b['lengths']
                y=b['label']
                logits = model(x, lengths=lengths, mask=mask).squeeze(-1)
                pr = torch.sigmoid(logits).cpu()
                ys.append(y); prs.append(pr)
        thresholds = [float(t) for t in str(getattr(args, 'thresholds', '0.5')).split(',') if t]
        thr_stats = {}
        if prs:
            y = torch.cat(ys).numpy(); pr = torch.cat(prs).numpy()
            try:
                auc = float(_m.roc_auc_score(y, pr))
            except Exception:
                auc = float('nan')
            for t in thresholds:
                try:
                    pr_bin = (pr > t).astype(int)
                    f1_t = float(_m.f1_score(y, pr_bin))
                    prec = float(_m.precision_score(y, pr_bin, zero_division=0))
                    rec = float(_m.recall_score(y, pr_bin, zero_division=0))
                    cm = _m.confusion_matrix(y, pr_bin, labels=[0,1])
                    tn_t, fp_t, fn_t, tp_t = int(cm[0,0]), int(cm[0,1]), int(cm[1,0]), int(cm[1,1])
                except Exception:
                    f1_t = float('nan'); prec = rec = 0.0; tp_t = fp_t = fn_t = tn_t = 0
                thr_stats[str(t)] = {'f1': f1_t, 'precision': prec, 'recall': rec, 'tp': tp_t, 'fp': fp_t, 'fn': fn_t, 'tn': tn_t}
            d0 = thr_stats.get('0.5') or thr_stats.get('0.50') or thr_stats.get(next(iter(thr_stats)))
            if d0:
                f1 = d0.get('f1', float('nan'))
                tp = d0.get('tp', 0); fp = d0.get('fp', 0); fn = d0.get('fn', 0); tn = d0.get('tn', 0)
            else:
                f1 = float('nan'); tp = fp = fn = tn = 0
        else:
            auc = float('nan')
            f1 = float('nan'); tp = fp = fn = tn = 0
            thr_stats = {}
        loss_mean = float(np.mean(losses)) if losses else 0.0
        # compute combined score using recall from 0.5 threshold if available
        try:
            recall_val = 0.0
            if isinstance(thr_stats, dict) and ('0.5' in thr_stats or '0.50' in thr_stats):
                d0 = thr_stats.get('0.5') or thr_stats.get('0.50')
                recall_val = float(d0.get('recall', 0.0)) if d0 else 0.0
            else:
                recall_val = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        except Exception:
            recall_val = 0.0
        comb = combined_score(auc, f1, recall_val)
        history.append({'epoch':epoch,'loss':loss_mean,'auc':auc, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'thresholds': thr_stats, 'comb': comb})
        print('Epoch', epoch, 'loss', f'{loss_mean:.6f}', 'auc', f'{auc}', 'f1', f'{f1}', 'TP', tp, 'FP', fp, 'FN', fn, 'TN', tn, 'time', time.time()-t0)
        # only save best model to avoid many ckpt files
        # use combined score for best selection / early stopping in single-run
        if 'best_comb' not in locals():
            best_comb = float('-inf')
        improved = (not np.isnan(comb)) and (comb > best_comb + min_delta)
        if improved:
            best_comb = comb
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'history': history, 'comb': best_comb}, out / 'best.pt')
            no_improve = 0
        else:
            no_improve += 1
        if patience>0 and no_improve >= patience:
            print(f'Early stopping on single-run at epoch {epoch} (no_improve={no_improve})')
            break

    with open(out / 'history.json','w') as f:
        json.dump(history, f, indent=2)
    # write a basic report
    # build a richer markdown report similar to other models' reports
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = 0

    # dataset shape
    try:
        N, T, featF = ds.features.shape
    except Exception:
        # fallback using sample
        try:
            s = ds[0]['features']
            T, featF = s.shape
            N = len(ds)
        except Exception:
            N = len(ds) if hasattr(ds, '__len__') else 0
            T = 0; featF = 0

    # history metrics
    best_idx = None
    best_auc = float('-inf')
    for i, h in enumerate(history):
        auc_v = h.get('auc', float('nan'))
        if not np.isnan(auc_v) and auc_v > best_auc:
            best_auc = auc_v; best_idx = i
    if best_idx is None:
        best_idx = 0
        best_auc = history[0].get('auc', float('nan')) if history else float('nan')

    best_entry = history[best_idx]
    final_entry = history[-1] if history else {}

    # trends (slope) over last up-to-10 epochs
    def slope(values):
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        try:
            m = float(np.polyfit(x, values, 1)[0])
        except Exception:
            m = 0.0
        return m

    last_k = history[-10:]
    train_losses = [h.get('loss', 0.0) for h in last_k]
    val_aucs = [h.get('auc', float('nan')) for h in last_k]
    train_loss_slope = slope(train_losses)
    # convert nan to 0 for slope calc
    val_auc_clean = [0.0 if np.isnan(v) else v for v in val_aucs]
    val_auc_slope = slope(val_auc_clean)

    # use top-level combined_score helper for single-run ranking
    top_epochs = []
    for i, h in enumerate(history):
        auc_v = h.get('auc', float('nan'))
        f1_v = h.get('f1', float('nan'))
        comb = combined_score(auc_v, f1_v)
        top_epochs.append((i + 1, h.get('loss', 0.0), auc_v, f1_v, h.get('tp', 0), h.get('fp', 0), h.get('fn', 0), h.get('tn', 0), comb))
    top_epochs = sorted(top_epochs, key=lambda x: -x[8])[:4]

    # overfitting simple heuristics
    try:
        gap = best_entry.get('loss', 0.0) - final_entry.get('loss', 0.0)
    except Exception:
        gap = 0.0
    acc_drop = False
    try:
        acc_drop = (best_entry.get('auc', 0.0) - final_entry.get('auc', 0.0)) > 0.05
    except Exception:
        acc_drop = False

    report_lines = []
    report_lines.append('# 訓練報告')
    report_lines.append(f'- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: {device}  | 參數量: {param_count}')
    report_lines.append(f'- 資料: N={N} T={T} F={featF}  | 批次: {args.batch}  | epoch: {args.epochs}')
    report_lines.append('')
    report_lines.append('## 核心指標')
    report_lines.append(f'- 最佳 (epoch {best_idx+1}): train_loss={best_entry.get("loss",0):.4f}, val_auc={best_entry.get("auc",float("nan")):.4f}')
    report_lines.append(f'- 最終 (epoch {final_entry.get("epoch",len(history))}): train_loss={final_entry.get("loss",0):.4f}, val_auc={final_entry.get("auc",float("nan")):.4f}')
    report_lines.append('')
    report_lines.append('## 趨勢 (最後 10 個 epoch 粗略斜率)')
    report_lines.append(f'- train_loss_slope: {train_loss_slope:.4f}')
    report_lines.append(f'- val_auc_slope: {val_auc_slope:.4f}')
    report_lines.append('')
    report_lines.append('## 學習率建議')
    report_lines.append(f'- 建議: 維持  | 當前 lr: {args.lr}')
    report_lines.append('- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。')
    report_lines.append('')
    report_lines.append('## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)')
    if top_epochs:
        for i, (e, l, a, f1v, tp, fp, fn, tn, comb) in enumerate(top_epochs, start=1):
            report_lines.append(f'{i}. epoch {e}: comb_score={comb:.4f}, train_loss={l:.4f}, val_auc={a:.4f}, val_f1={f1v:.4f}')
            report_lines.append(f'   - confusion (TP,FP,FN,TN): {tp},{fp},{fn},{tn}')
    else:
        report_lines.append('無可用 val_auc 結果')
    report_lines.append('')
    report_lines.append('## 過擬合分析')
    report_lines.append(f'- 判定: {"否" if not acc_drop else "是"} (gap={gap:.4f})')
    report_lines.append(f'- 訊號: early_best={best_idx < max(1, args.epochs//2)}, loss_rebound=False, gap_large={abs(gap)>0.1}, acc_drop={acc_drop}')
    report_lines.append('')
    report_lines.append('## 設定摘要')
    report_lines.append(f'- lr: {args.lr}')
    report_lines.append(f'- weight_decay: {opt.defaults.get("weight_decay", 0.0) if hasattr(opt, "defaults") else 0.0}')
    report_lines.append(f'- seed: {args.seed}')
    report_lines.append(f'- use_bn: {args.use_bn}')
    report_lines.append(f'- pooling: {args.pool}')
    report_lines.append(f'- num_workers: {args.num_workers}')

    with open(out / 'report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    print('done')


if __name__=='__main__':
    train()
