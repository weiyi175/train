# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed49 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 49 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7328
- F1: 0.7325
- Recall: 0.8261
- Precision: 0.6580
- Composite Score: 0.7794 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6953 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 152
- FP: 79
- FN: 32
- TN: 91

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7821, f1=0.7132, recall=0.8679, precision=0.6053, score=0.8043, precisionAware=0.6730  TP:46 FP:30 FN:7 TN:45
- epoch 54: auc=0.7824, f1=0.6923, recall=0.8491, precision=0.5844, score=0.7887, precisionAware=0.6564  TP:45 FP:32 FN:8 TN:43
- epoch 58: auc=0.7882, f1=0.7107, recall=0.8113, precision=0.6324, score=0.7765, precisionAware=0.6870  TP:43 FP:25 FN:10 TN:50
- epoch 55: auc=0.7879, f1=0.7119, recall=0.7925, precision=0.6462, score=0.7674, precisionAware=0.6942  TP:42 FP:23 FN:11 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7960, f1=0.7130, recall=0.7736, precision=0.6613, precisionAware=0.7038, composite=0.7599  TP:41 FP:21 FN:12 TN:54
- epoch 55: auc=0.7879, f1=0.7119, recall=0.7925, precision=0.6462, precisionAware=0.6942, composite=0.7674  TP:42 FP:23 FN:11 TN:52
- epoch 56: auc=0.7960, f1=0.6847, recall=0.7170, precision=0.6552, precisionAware=0.6922, composite=0.7231  TP:38 FP:20 FN:15 TN:55
- epoch 52: auc=0.7897, f1=0.7009, recall=0.7736, precision=0.6406, precisionAware=0.6885, composite=0.7550  TP:41 FP:23 FN:12 TN:52
