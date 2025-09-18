# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10
```

## Test metrics
- AUC: 0.5917
- F1: 0.5149
- Recall: 0.4952
## Confusion matrix (TP/FP/FN/TN)
- TP: 52
- FP: 45
- FN: 53
- TN: 106

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 70: auc=0.6762, f1=0.5474, recall=0.4906, score=0.5633  TP: 26 FP: 16 FN: 27 TN: 59
- epoch 80: auc=0.6762, f1=0.5474, recall=0.4906, score=0.5633  TP: 26 FP: 16 FN: 27 TN: 59
- epoch 71: auc=0.6755, f1=0.5361, recall=0.4906, score=0.5597  TP: 26 FP: 18 FN: 27 TN: 57
- epoch 68: auc=0.6684, f1=0.5111, recall=0.4340, score=0.5274  TP: 23 FP: 14 FN: 30 TN: 61
