# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type auto --arch tcn --attn_units 32 --gate_type vector_sigmoid --epochs 1 --batch 8 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10 --no_early_stop --run_seed None
```

## Test metrics
- AUC: 0.6169
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 1: auc=0.6181, f1=0.0000, recall=0.0000, score=0.1854  TP: 0 FP: 0 FN: 53 TN: 75
