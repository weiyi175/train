# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 14
```

## Test metrics
- AUC: 0.6040
- F1: 0.1379
- Recall: 0.0762
## Confusion matrix (TP/FP/FN/TN)
- TP: 8
- FP: 3
- FN: 97
- TN: 148

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 28: auc=0.6460, f1=0.3733, recall=0.2642, score=0.4115  TP: 14 FP: 8 FN: 39 TN: 67
- epoch 26: auc=0.6481, f1=0.2687, recall=0.1698, score=0.3429  TP: 9 FP: 5 FN: 44 TN: 70
- epoch 27: auc=0.6385, f1=0.2388, recall=0.1509, score=0.3236  TP: 8 FP: 6 FN: 45 TN: 69
- epoch 25: auc=0.6501, f1=0.2188, recall=0.1321, score=0.3135  TP: 7 FP: 4 FN: 46 TN: 71
