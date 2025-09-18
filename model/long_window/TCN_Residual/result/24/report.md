# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 22
```

## Test metrics
- AUC: 0.6382
- F1: 0.2276
- Recall: 0.1333
## Confusion matrix (TP/FP/FN/TN)
- TP: 14
- FP: 4
- FN: 91
- TN: 147

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 18: auc=0.6272, f1=0.3429, recall=0.2264, score=0.3816  TP: 12 FP: 5 FN: 41 TN: 70
- epoch 21: auc=0.6231, f1=0.3333, recall=0.2264, score=0.3775  TP: 12 FP: 7 FN: 41 TN: 68
- epoch 20: auc=0.6171, f1=0.3333, recall=0.2264, score=0.3757  TP: 12 FP: 7 FN: 41 TN: 68
- epoch 19: auc=0.6244, f1=0.2727, recall=0.1698, score=0.3371  TP: 9 FP: 4 FN: 44 TN: 71
