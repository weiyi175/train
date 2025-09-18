# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 16 --gate_type sigmoid --epochs 60 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10
```

## Test metrics
- AUC: 0.5370
- F1: 0.4086
- Recall: 0.3619
## Confusion matrix (TP/FP/FN/TN)
- TP: 38
- FP: 43
- FN: 67
- TN: 108

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 40: auc=0.5917, f1=0.4235, recall=0.3396, score=0.4404  TP: 18 FP: 14 FN: 35 TN: 61
- epoch 49: auc=0.5995, f1=0.4045, recall=0.3396, score=0.4370  TP: 18 FP: 18 FN: 35 TN: 57
- epoch 44: auc=0.5925, f1=0.4091, recall=0.3396, score=0.4363  TP: 18 FP: 17 FN: 35 TN: 58
- epoch 48: auc=0.6073, f1=0.4048, recall=0.3208, score=0.4319  TP: 17 FP: 14 FN: 36 TN: 61
