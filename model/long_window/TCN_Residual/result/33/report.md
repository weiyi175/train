# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 1
```

## Test metrics
- AUC: 0.5856
- F1: 0.3077
- Recall: 0.2095
## Confusion matrix (TP/FP/FN/TN)
- TP: 22
- FP: 16
- FN: 83
- TN: 135

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 29: auc=0.6133, f1=0.3571, recall=0.2830, score=0.4044  TP: 15 FP: 16 FN: 38 TN: 59
- epoch 31: auc=0.6111, f1=0.3571, recall=0.2830, score=0.4037  TP: 15 FP: 16 FN: 38 TN: 59
- epoch 33: auc=0.6048, f1=0.3571, recall=0.2830, score=0.4018  TP: 15 FP: 16 FN: 38 TN: 59
- epoch 28: auc=0.6116, f1=0.3373, recall=0.2642, score=0.3903  TP: 14 FP: 16 FN: 39 TN: 59
