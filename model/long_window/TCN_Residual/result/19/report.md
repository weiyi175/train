# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed 22
```

## Test metrics
- AUC: 0.6079
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 15: auc=0.5240, f1=0.0968, recall=0.0566, score=0.2089  TP: 3 FP: 6 FN: 50 TN: 69
- epoch 14: auc=0.5268, f1=0.0667, recall=0.0377, score=0.1931  TP: 2 FP: 5 FN: 51 TN: 70
- epoch 11: auc=0.5424, f1=0.0357, recall=0.0189, score=0.1810  TP: 1 FP: 2 FN: 52 TN: 73
- epoch 13: auc=0.5333, f1=0.0357, recall=0.0189, score=0.1783  TP: 1 FP: 2 FN: 52 TN: 73
