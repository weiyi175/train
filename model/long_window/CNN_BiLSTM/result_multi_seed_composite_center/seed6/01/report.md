# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed6 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 6 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7035
- F1: 0.6649
- Recall: 0.6739
- Precision: 0.6561
- Composite Score: 0.6771 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6682 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 124
- FP: 65
- FN: 60
- TN: 105

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 26: auc=0.7798, f1=0.6981, recall=0.6852, precision=0.7115, score=0.7080, precisionAware=0.7212  TP:37 FP:15 FN:17 TN:59
- epoch 70: auc=0.7723, f1=0.6552, recall=0.7037, precision=0.6129, score=0.7029, precisionAware=0.6575  TP:38 FP:24 FN:16 TN:50
- epoch 64: auc=0.7670, f1=0.6667, recall=0.6852, precision=0.6491, score=0.6960, precisionAware=0.6780  TP:37 FP:20 FN:17 TN:54
- epoch 67: auc=0.7513, f1=0.6435, recall=0.6852, precision=0.6066, score=0.6859, precisionAware=0.6466  TP:37 FP:24 FN:17 TN:50

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 19: auc=0.7778, f1=0.6207, recall=0.5000, precision=0.8182, precisionAware=0.7509, composite=0.5918  TP:27 FP:6 FN:27 TN:68
- epoch 17: auc=0.7578, f1=0.6526, recall=0.5741, precision=0.7561, precisionAware=0.7254, composite=0.6344  TP:31 FP:10 FN:23 TN:64
- epoch 20: auc=0.7765, f1=0.6667, recall=0.6111, precision=0.7333, precisionAware=0.7220, composite=0.6609  TP:33 FP:12 FN:21 TN:62
- epoch 26: auc=0.7798, f1=0.6981, recall=0.6852, precision=0.7115, precisionAware=0.7212, composite=0.7080  TP:37 FP:15 FN:17 TN:59
