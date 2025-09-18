# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.5 --focal_gamma_end 0.5 --curriculum_epochs 1 --run_seed 25
```

## Test metrics
- AUC: 0.5881
- F1: 0.1951
- Recall: 0.1143
## Confusion matrix (TP/FP/FN/TN)
- TP: 12
- FP: 6
- FN: 93
- TN: 145

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 25: auc=0.5844, f1=0.3188, recall=0.2075, score=0.3540  TP: 11 FP: 5 FN: 42 TN: 70
- epoch 24: auc=0.5839, f1=0.2857, recall=0.1887, score=0.3364  TP: 10 FP: 7 FN: 43 TN: 68
- epoch 26: auc=0.5761, f1=0.2857, recall=0.1887, score=0.3340  TP: 10 FP: 7 FN: 43 TN: 68
- epoch 27: auc=0.5701, f1=0.2817, recall=0.1887, score=0.3310  TP: 10 FP: 8 FN: 43 TN: 67
