# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.5 --focal_gamma_end 0.5 --curriculum_epochs 1 --run_seed 23
```

## Test metrics
- AUC: 0.6220
- F1: 0.4026
- Recall: 0.2952
## Confusion matrix (TP/FP/FN/TN)
- TP: 31
- FP: 18
- FN: 74
- TN: 133

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 30: auc=0.5844, f1=0.4096, recall=0.3208, score=0.4265  TP: 17 FP: 13 FN: 36 TN: 62
- epoch 29: auc=0.5839, f1=0.3846, recall=0.2830, score=0.4038  TP: 15 FP: 10 FN: 38 TN: 65
- epoch 25: auc=0.5814, f1=0.3200, recall=0.2264, score=0.3610  TP: 12 FP: 10 FN: 41 TN: 65
- epoch 26: auc=0.5796, f1=0.3077, recall=0.2264, score=0.3568  TP: 12 FP: 13 FN: 41 TN: 62
