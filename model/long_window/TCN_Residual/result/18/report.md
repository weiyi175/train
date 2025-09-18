# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type long --arch gated --attn_units 32 --gate_type vector_sigmoid --epochs 100 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --use_layernorm --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed 21
```

## Test metrics
- AUC: 0.6400
- F1: 0.1920
- Recall: 0.1143
## Confusion matrix (TP/FP/FN/TN)
- TP: 12
- FP: 8
- FN: 93
- TN: 143

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 18: auc=0.5389, f1=0.2535, recall=0.1698, score=0.3056  TP: 9 FP: 9 FN: 44 TN: 66
- epoch 22: auc=0.5240, f1=0.2432, recall=0.1698, score=0.2981  TP: 9 FP: 12 FN: 44 TN: 63
- epoch 19: auc=0.5346, f1=0.2319, recall=0.1509, score=0.2903  TP: 8 FP: 8 FN: 45 TN: 67
- epoch 20: auc=0.5346, f1=0.2222, recall=0.1509, score=0.2874  TP: 8 FP: 11 FN: 45 TN: 64
