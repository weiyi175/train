# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 64 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10
```

## Test metrics
- AUC: 0.5921
- F1: 0.4330
- Recall: 0.4000
## Confusion matrix (TP/FP/FN/TN)
- TP: 42
- FP: 47
- FN: 63
- TN: 104

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 49: auc=0.5265, f1=0.4086, recall=0.3585, score=0.4239  TP: 19 FP: 21 FN: 34 TN: 54
- epoch 42: auc=0.5238, f1=0.3958, recall=0.3585, score=0.4193  TP: 19 FP: 24 FN: 34 TN: 51
- epoch 51: auc=0.5278, f1=0.3956, recall=0.3396, score=0.4129  TP: 18 FP: 20 FN: 35 TN: 55
- epoch 50: auc=0.5240, f1=0.3956, recall=0.3396, score=0.4117  TP: 18 FP: 20 FN: 35 TN: 55
