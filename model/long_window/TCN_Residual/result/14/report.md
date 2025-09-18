# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.5 --focal_gamma_end 0.5 --curriculum_epochs 1 --run_seed 17
```

## Test metrics
- AUC: 0.5019
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 1
- FN: 105
- TN: 150

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 1: auc=0.6035, f1=0.0364, recall=0.0189, score=0.1995  TP: 1 FP: 1 FN: 52 TN: 74
- epoch 11: auc=0.6035, f1=0.0364, recall=0.0189, score=0.1995  TP: 1 FP: 1 FN: 52 TN: 74
- epoch 3: auc=0.5884, f1=0.0364, recall=0.0189, score=0.1950  TP: 1 FP: 1 FN: 52 TN: 74
- epoch 6: auc=0.5877, f1=0.0364, recall=0.0189, score=0.1948  TP: 1 FP: 1 FN: 52 TN: 74
