# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 22
```

## Test metrics
- AUC: 0.6562
- F1: 0.1053
- Recall: 0.0571
## Confusion matrix (TP/FP/FN/TN)
- TP: 6
- FP: 3
- FN: 99
- TN: 148

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 11: auc=0.5348, f1=0.1587, recall=0.0943, score=0.2458  TP: 5 FP: 5 FN: 48 TN: 70
- epoch 15: auc=0.5341, f1=0.1493, recall=0.0943, score=0.2427  TP: 5 FP: 9 FN: 48 TN: 66
- epoch 12: auc=0.5215, f1=0.1562, recall=0.0943, score=0.2411  TP: 5 FP: 6 FN: 48 TN: 69
- epoch 14: auc=0.5233, f1=0.1515, recall=0.0943, score=0.2402  TP: 5 FP: 8 FN: 48 TN: 67
