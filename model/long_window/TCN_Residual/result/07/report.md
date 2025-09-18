# TCN_Residual Report

## Command
```
python train_tcn_residual.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --window_type auto --arch gated --attn_units None --gate_type vector_sigmoid --epochs 3 --batch 64 --filters 64 --kernel 3 --dropout 0.2 --result_dir /home/user/projects/train/model/long_window/TCN_Residual/result --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 10
```

## Test metrics
- AUC: 0.5760
- F1: 0.0183
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 3
- FN: 104
- TN: 148

## Top 4 epochs by score = 0.4*recall + 0.3*f1 + 0.3*auc
- epoch 3: auc=0.6206, f1=0.0727, recall=0.0377, score=0.2231  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 1: auc=0.6141, f1=0.0727, recall=0.0377, score=0.2211  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 2: auc=0.6156, f1=0.0370, recall=0.0189, score=0.2033  TP: 1 FP: 0 FN: 52 TN: 75
