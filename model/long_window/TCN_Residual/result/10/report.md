# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 10 --run_seed 13
```

## Test metrics
- AUC: 0.5956
- F1: 0.3077
- Recall: 0.2095
## Confusion matrix (TP/FP/FN/TN)
- TP: 22
- FP: 16
- FN: 83
- TN: 135

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 33: auc=0.5927, f1=0.3099, recall=0.2075, score=0.3538  TP: 11 FP: 7 FN: 42 TN: 68
- epoch 32: auc=0.5914, f1=0.3099, recall=0.2075, score=0.3534  TP: 11 FP: 7 FN: 42 TN: 68
- epoch 35: auc=0.5857, f1=0.3014, recall=0.2075, score=0.3491  TP: 11 FP: 9 FN: 42 TN: 66
- epoch 34: auc=0.5864, f1=0.2857, recall=0.1887, score=0.3371  TP: 10 FP: 7 FN: 43 TN: 68
