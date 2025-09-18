# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 1.0 --focal_gamma_end 1.0 --curriculum_epochs 1 --run_seed 18
```

## Test metrics
- AUC: 0.4675
- F1: 0.1875
- Recall: 0.1143
## Confusion matrix (TP/FP/FN/TN)
- TP: 12
- FP: 11
- FN: 93
- TN: 140

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 1: auc=0.5072, f1=0.1846, recall=0.1132, score=0.2528  TP: 6 FP: 6 FN: 47 TN: 69
- epoch 11: auc=0.5072, f1=0.1846, recall=0.1132, score=0.2528  TP: 6 FP: 6 FN: 47 TN: 69
- epoch 2: auc=0.4941, f1=0.0690, recall=0.0377, score=0.1840  TP: 2 FP: 3 FN: 51 TN: 72
- epoch 3: auc=0.4803, f1=0.0678, recall=0.0377, score=0.1795  TP: 2 FP: 4 FN: 51 TN: 71
