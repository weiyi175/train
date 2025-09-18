# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.5 --focal_gamma_end 0.5 --curriculum_epochs 1 --run_seed 23
```

## Test metrics
- AUC: 0.6312
- F1: 0.2941
- Recall: 0.1905
## Confusion matrix (TP/FP/FN/TN)
- TP: 20
- FP: 11
- FN: 85
- TN: 140

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 21: auc=0.4642, f1=0.2564, recall=0.1887, score=0.2916  TP: 10 FP: 15 FN: 43 TN: 60
- epoch 20: auc=0.4699, f1=0.2368, recall=0.1698, score=0.2800  TP: 9 FP: 14 FN: 44 TN: 61
- epoch 18: auc=0.4639, f1=0.2308, recall=0.1698, score=0.2763  TP: 9 FP: 16 FN: 44 TN: 59
- epoch 19: auc=0.4624, f1=0.2278, recall=0.1698, score=0.2750  TP: 9 FP: 17 FN: 44 TN: 58
