# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --no_early_stop --run_seed 24
```

## Test metrics
- AUC: 0.5294
- F1: 0.3253
- Recall: 0.2571
## Confusion matrix (TP/FP/FN/TN)
- TP: 27
- FP: 34
- FN: 78
- TN: 117

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 49: auc=0.5842, f1=0.4800, recall=0.4528, score=0.5004  TP: 24 FP: 23 FN: 29 TN: 52
- epoch 96: auc=0.5804, f1=0.4706, recall=0.4528, score=0.4964  TP: 24 FP: 25 FN: 29 TN: 50
- epoch 95: auc=0.5784, f1=0.4706, recall=0.4528, score=0.4958  TP: 24 FP: 25 FN: 29 TN: 50
- epoch 86: auc=0.5761, f1=0.4706, recall=0.4528, score=0.4951  TP: 24 FP: 25 FN: 29 TN: 50
