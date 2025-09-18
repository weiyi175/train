# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.2 --focal_gamma_end 1.2 --curriculum_epochs 20 --run_seed 19
```

## Test metrics
- AUC: 0.6224
- F1: 0.2979
- Recall: 0.2000
## Confusion matrix (TP/FP/FN/TN)
- TP: 21
- FP: 15
- FN: 84
- TN: 136

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 28: auc=0.5867, f1=0.3333, recall=0.2453, score=0.3741  TP: 13 FP: 12 FN: 40 TN: 63
- epoch 29: auc=0.5930, f1=0.3158, recall=0.2264, score=0.3632  TP: 12 FP: 11 FN: 41 TN: 64
- epoch 27: auc=0.5844, f1=0.3158, recall=0.2264, score=0.3606  TP: 12 FP: 11 FN: 41 TN: 64
- epoch 26: auc=0.5892, f1=0.2895, recall=0.2075, score=0.3466  TP: 11 FP: 12 FN: 42 TN: 63
