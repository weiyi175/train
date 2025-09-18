# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 16
```

## Test metrics
- AUC: 0.5693
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 11: auc=0.5182, f1=0.0984, recall=0.0566, score=0.2076  TP: 3 FP: 5 FN: 50 TN: 70
- epoch 2: auc=0.5389, f1=0.0000, recall=0.0000, score=0.1617  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 12: auc=0.5389, f1=0.0000, recall=0.0000, score=0.1617  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 1: auc=0.5341, f1=0.0000, recall=0.0000, score=0.1602  TP: 0 FP: 0 FN: 53 TN: 75
