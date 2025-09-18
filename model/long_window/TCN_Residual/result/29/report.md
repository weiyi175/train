# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 5 --run_seed 26
```

## Test metrics
- AUC: 0.5881
- F1: 0.2273
- Recall: 0.1429
## Confusion matrix (TP/FP/FN/TN)
- TP: 15
- FP: 12
- FN: 90
- TN: 139

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 24: auc=0.5263, f1=0.1944, recall=0.1321, score=0.2691  TP: 7 FP: 12 FN: 46 TN: 63
- epoch 17: auc=0.5512, f1=0.1667, recall=0.1132, score=0.2606  TP: 6 FP: 13 FN: 47 TN: 62
- epoch 19: auc=0.5487, f1=0.1667, recall=0.1132, score=0.2599  TP: 6 FP: 13 FN: 47 TN: 62
- epoch 21: auc=0.5318, f1=0.1765, recall=0.1132, score=0.2578  TP: 6 FP: 9 FN: 47 TN: 66
