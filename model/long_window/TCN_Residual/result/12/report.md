# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 10 --run_seed 15
```

## Test metrics
- AUC: 0.5360
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 7: auc=0.5371, f1=0.1587, recall=0.0943, score=0.2465  TP: 5 FP: 5 FN: 48 TN: 70
- epoch 8: auc=0.5288, f1=0.1562, recall=0.0943, score=0.2433  TP: 5 FP: 6 FN: 48 TN: 69
- epoch 9: auc=0.5255, f1=0.1562, recall=0.0943, score=0.2423  TP: 5 FP: 6 FN: 48 TN: 69
- epoch 10: auc=0.5220, f1=0.1562, recall=0.0943, score=0.2412  TP: 5 FP: 6 FN: 48 TN: 69
