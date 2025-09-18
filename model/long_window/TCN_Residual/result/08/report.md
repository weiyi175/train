# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10 --run_seed 11
```

## Test metrics
- AUC: 0.6462
- F1: 0.3143
- Recall: 0.2095
## Confusion matrix (TP/FP/FN/TN)
- TP: 22
- FP: 13
- FN: 83
- TN: 138

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 31: auc=0.5839, f1=0.3467, recall=0.2453, score=0.3773  TP: 13 FP: 9 FN: 40 TN: 66
- epoch 33: auc=0.5786, f1=0.3421, recall=0.2453, score=0.3743  TP: 13 FP: 10 FN: 40 TN: 65
- epoch 30: auc=0.5852, f1=0.3333, recall=0.2264, score=0.3661  TP: 12 FP: 7 FN: 41 TN: 68
- epoch 32: auc=0.5847, f1=0.3243, recall=0.2264, score=0.3633  TP: 12 FP: 9 FN: 41 TN: 66
