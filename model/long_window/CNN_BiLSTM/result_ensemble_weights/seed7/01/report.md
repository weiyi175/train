# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed7 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 7 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6989
- F1: 0.6597
- Recall: 0.6848
- Precision: 0.6364
- Composite Score: 0.6801 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6559 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 126
- FP: 72
- FN: 58
- TN: 98

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7542, f1=0.6500, recall=0.7358, precision=0.5821, score=0.7138, precisionAware=0.6369  TP:39 FP:28 FN:14 TN:47
- epoch 51: auc=0.7472, f1=0.6609, recall=0.7170, precision=0.6129, score=0.7062, precisionAware=0.6541  TP:38 FP:24 FN:15 TN:51
- epoch 55: auc=0.7316, f1=0.6341, recall=0.7358, precision=0.5571, score=0.7045, precisionAware=0.6151  TP:39 FP:31 FN:14 TN:44
- epoch 49: auc=0.7323, f1=0.6496, recall=0.7170, precision=0.5938, score=0.6998, precisionAware=0.6382  TP:38 FP:26 FN:15 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 51: auc=0.7472, f1=0.6609, recall=0.7170, precision=0.6129, precisionAware=0.6541, composite=0.7062  TP:38 FP:24 FN:15 TN:51
- epoch 59: auc=0.7550, f1=0.6491, recall=0.6981, precision=0.6066, precisionAware=0.6490, composite=0.6948  TP:37 FP:24 FN:16 TN:51
- epoch 58: auc=0.7469, f1=0.6491, recall=0.6981, precision=0.6066, precisionAware=0.6474, composite=0.6932  TP:37 FP:24 FN:16 TN:51
- epoch 54: auc=0.7439, f1=0.6372, recall=0.6792, precision=0.6000, precisionAware=0.6399, composite=0.6796  TP:36 FP:24 FN:17 TN:51
