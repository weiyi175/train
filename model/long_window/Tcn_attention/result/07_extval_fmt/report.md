# TCN_Attention_Long Report

## Command
```
/home/user/projects/train/model/long_window/Tcn_attention/train_tcn_attn_long.py --train_npz /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_npz /home/user/projects/train/Val_data/slipce_thresh040/windows_npz.npz --test_npz /home/user/projects/train/test_data/slipce_thresh040/windows_npz.npz --epochs 30 --batch_size 32 --lr 1e-4 --use_norm --balance_by_class --hard_negative_factor 1.0 --early_stop_patience 0 --tag extval_fmt
```

## Test metrics
- AUC: 0.6464
- F1: 0.5241
- Recall: 0.4028
- Precision: 0.7500
- Composite Score: 0.4879 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6615 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 87
- FP: 29
- FN: 129
- TN: 109

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 4: auc=0.5771, f1=0.8293, recall=0.9273, precision=0.7500, score=0.8278, precisionAware=0.7392  TP:102 FP:34 FN:8 TN:8
- epoch 18: auc=0.5303, f1=0.7950, recall=0.8636, precision=0.7364, score=0.7764, precisionAware=0.7128  TP:95 FP:34 FN:15 TN:8
- epoch 13: auc=0.5695, f1=0.7826, recall=0.8182, precision=0.7500, score=0.7578, precisionAware=0.7237  TP:90 FP:30 FN:20 TN:12
- epoch 6: auc=0.5554, f1=0.7911, recall=0.8091, precision=0.7739, score=0.7530, precisionAware=0.7354  TP:89 FP:26 FN:21 TN:16

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 8: auc=0.6102, f1=0.7727, recall=0.7727, precision=0.7727, precisionAware=0.7402, composite=0.7402  TP:85 FP:25 FN:25 TN:17
- epoch 4: auc=0.5771, f1=0.8293, recall=0.9273, precision=0.7500, precisionAware=0.7392, composite=0.8278  TP:102 FP:34 FN:8 TN:8
- epoch 6: auc=0.5554, f1=0.7911, recall=0.8091, precision=0.7739, precisionAware=0.7354, composite=0.7530  TP:89 FP:26 FN:21 TN:16
- epoch 5: auc=0.6134, f1=0.6907, recall=0.6091, precision=0.7976, precisionAware=0.7287, composite=0.6344  TP:67 FP:17 FN:43 TN:25
