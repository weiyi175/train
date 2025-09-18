# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed 12
```

## Test metrics
- AUC: 0.6176
- F1: 0.2406
- Recall: 0.1524
## Confusion matrix (TP/FP/FN/TN)
- TP: 16
- FP: 12
- FN: 89
- TN: 139

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 13: auc=0.5600, f1=0.2727, recall=0.1698, score=0.3177  TP: 9 FP: 4 FN: 44 TN: 71
- epoch 15: auc=0.5519, f1=0.2687, recall=0.1698, score=0.3141  TP: 9 FP: 5 FN: 44 TN: 70
- epoch 14: auc=0.5487, f1=0.2687, recall=0.1698, score=0.3131  TP: 9 FP: 5 FN: 44 TN: 70
- epoch 17: auc=0.5512, f1=0.2535, recall=0.1698, score=0.3093  TP: 9 FP: 9 FN: 44 TN: 66
