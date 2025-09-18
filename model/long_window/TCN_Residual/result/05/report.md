# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type auto --arch tcn --attn_units None --gate_type sigmoid --epochs 3 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10
```

## Test metrics
- AUC: 0.5956
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 3: auc=0.5595, f1=0.0370, recall=0.0189, score=0.1865  TP: 1 FP: 0 FN: 52 TN: 75
- epoch 2: auc=0.5386, f1=0.0000, recall=0.0000, score=0.1616  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 1: auc=0.5338, f1=0.0000, recall=0.0000, score=0.1602  TP: 0 FP: 0 FN: 53 TN: 75
