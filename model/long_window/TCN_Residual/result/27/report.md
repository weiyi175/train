# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 24
```

## Test metrics
- AUC: 0.5396
- F1: 0.0370
- Recall: 0.0190
## Confusion matrix (TP/FP/FN/TN)
- TP: 2
- FP: 1
- FN: 103
- TN: 150

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 14: auc=0.5653, f1=0.1034, recall=0.0566, score=0.2233  TP: 3 FP: 2 FN: 50 TN: 73
- epoch 15: auc=0.5638, f1=0.1000, recall=0.0566, score=0.2218  TP: 3 FP: 4 FN: 50 TN: 71
- epoch 12: auc=0.5653, f1=0.0727, recall=0.0377, score=0.2065  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 13: auc=0.5665, f1=0.0690, recall=0.0377, score=0.2057  TP: 2 FP: 3 FN: 51 TN: 72
