# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 15 --run_seed 20
```

## Test metrics
- AUC: 0.6145
- F1: 0.2443
- Recall: 0.1524
## Confusion matrix (TP/FP/FN/TN)
- TP: 16
- FP: 10
- FN: 89
- TN: 141

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 15: auc=0.5965, f1=0.3478, recall=0.2264, score=0.3739  TP: 12 FP: 4 FN: 41 TN: 71
- epoch 16: auc=0.5894, f1=0.3188, recall=0.2075, score=0.3555  TP: 11 FP: 5 FN: 42 TN: 70
- epoch 17: auc=0.5937, f1=0.2941, recall=0.1887, score=0.3418  TP: 10 FP: 5 FN: 43 TN: 70
- epoch 18: auc=0.5864, f1=0.2899, recall=0.1887, score=0.3384  TP: 10 FP: 6 FN: 43 TN: 69
