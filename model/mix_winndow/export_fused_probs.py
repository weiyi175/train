#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from mix_dataset import OfflineFusionDataset
from models_mixed import FUSION_BUILDERS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion-dir', required=True, help='目標融合訓練輸出資料夾 (包含 fusion.ckpt, threshold_best.json)')
    ap.add_argument('--fusion', choices=['weighted','mlp','stack_logistic'], default='mlp')
    ap.add_argument('--short-probs', required=True)
    ap.add_argument('--long-probs', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--threshold', type=float, default=None, help='若指定則覆蓋 threshold_best.json 中的門檻')
    ap.add_argument('--which-threshold', choices=['best_composite','best_precision_aware'], default='best_precision_aware')
    ap.add_argument('--out-name', default='fused_probs.npy')
    return ap.parse_args()


def evaluate(probs, y, thr):
    from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix
    preds = (probs >= thr).astype(int)
    if len(np.unique(y))>1:
        try: auc=float(roc_auc_score(y, probs))
        except Exception: auc=0.0
    else: auc=0.0
    f1=float(f1_score(y,preds,zero_division=0))
    recall=float(recall_score(y,preds,zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y,preds).ravel() if len(np.unique(preds))==2 else (0,0,0,0)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    composite = 0.5*recall + 0.3*f1 + 0.2*auc
    precision_aware = 0.5*precision + 0.3*f1 + 0.2*auc
    return dict(threshold=thr,auc=auc,f1=f1,recall=recall,precision=precision,composite=composite,precision_aware=precision_aware,TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn))


def main():
    args = parse_args()
    fusion_dir=Path(args.fusion_dir)
    ckpt_path=fusion_dir/'fusion.ckpt'
    best_path=fusion_dir/'threshold_best.json'
    assert ckpt_path.exists(), f'缺少 {ckpt_path}'
    if args.threshold is None:
        data=json.load(best_path.open())
        sel=data[args.which_threshold]['threshold']
    else:
        sel=args.threshold
    print('[INFO] 使用門檻:', sel, '(source:', 'arg' if args.threshold is not None else args.which_threshold, ')')

    # 準備資料
    ds=OfflineFusionDataset(args.short_probs,args.long_probs,args.labels)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=FUSION_BUILDERS[args.fusion]()
    state=torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model'])
    model.to(device).eval()

    # 推論
    loader=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=False)
    all=[]
    with torch.no_grad():
        for b in loader:
            p=torch.stack([b['p_short'], b['p_long']],dim=1).to(device).float()
            logits=model(p)
            prob=torch.softmax(logits,dim=-1)[:,1].cpu().numpy()
            all.append(prob)
    probs=np.concatenate(all,0).astype('float32')

    # 評估
    metrics=evaluate(probs, ds.labels, float(sel))
    out_prob=fusion_dir/args.out_name
    np.save(out_prob, probs)
    json.dump({'selected_threshold': float(sel), 'which': args.which_threshold, 'metrics': metrics}, open(fusion_dir/'selected_threshold.json','w'), ensure_ascii=False, indent=2)
    print('Saved:', out_prob, 'metrics:', metrics)

if __name__=='__main__':
    main()
