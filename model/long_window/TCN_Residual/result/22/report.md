# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 22
```

## Test metrics
- AUC: 0.6013
- F1: 0.2781
- Recall: 0.2000
## Confusion matrix (TP/FP/FN/TN)
- TP: 21
- FP: 25
- FN: 84
- TN: 126

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 42: auc=0.6126, f1=0.4048, recall=0.3208, score=0.4335  TP: 17 FP: 14 FN: 36 TN: 61
- epoch 40: auc=0.6169, f1=0.3765, recall=0.3019, score=0.4188  TP: 16 FP: 16 FN: 37 TN: 59
- epoch 44: auc=0.6083, f1=0.3810, recall=0.3019, score=0.4175  TP: 16 FP: 15 FN: 37 TN: 60
- epoch 43: auc=0.6058, f1=0.3659, recall=0.2830, score=0.4047  TP: 15 FP: 14 FN: 38 TN: 61
