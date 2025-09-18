# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --no_early_stop --run_seed 24
```

## Test metrics
- AUC: 0.5849
- F1: 0.1176
- Recall: 0.0667
## Confusion matrix (TP/FP/FN/TN)
- TP: 7
- FP: 7
- FN: 98
- TN: 144

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 67: auc=0.4933, f1=0.4301, recall=0.3774, score=0.4280  TP: 20 FP: 20 FN: 33 TN: 55
- epoch 48: auc=0.5155, f1=0.4176, recall=0.3585, score=0.4233  TP: 19 FP: 19 FN: 34 TN: 56
- epoch 43: auc=0.5162, f1=0.4286, recall=0.3396, score=0.4193  TP: 18 FP: 13 FN: 35 TN: 62
- epoch 60: auc=0.4969, f1=0.4222, recall=0.3585, score=0.4191  TP: 19 FP: 18 FN: 34 TN: 57
