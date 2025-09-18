# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed 21
```

## Test metrics
- AUC: 0.5857
- F1: 0.1818
- Recall: 0.1143
## Confusion matrix (TP/FP/FN/TN)
- TP: 12
- FP: 15
- FN: 93
- TN: 136

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 34: auc=0.5824, f1=0.3467, recall=0.2453, score=0.3768  TP: 13 FP: 9 FN: 40 TN: 66
- epoch 32: auc=0.5821, f1=0.2973, recall=0.2075, score=0.3468  TP: 11 FP: 10 FN: 42 TN: 65
- epoch 33: auc=0.5824, f1=0.2817, recall=0.1887, score=0.3347  TP: 10 FP: 8 FN: 43 TN: 67
- epoch 31: auc=0.5806, f1=0.2254, recall=0.1509, score=0.3022  TP: 8 FP: 10 FN: 45 TN: 65
