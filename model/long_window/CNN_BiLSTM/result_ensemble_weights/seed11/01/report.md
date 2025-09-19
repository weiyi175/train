# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed11 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 11 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7125
- F1: 0.6519
- Recall: 0.6413
- Precision: 0.6629
- Composite Score: 0.6587 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6696 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 118
- FP: 60
- FN: 66
- TN: 110

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7042, f1=0.6290, recall=0.7358, precision=0.5493, score=0.6975, precisionAware=0.6042  TP:39 FP:32 FN:14 TN:43
- epoch 53: auc=0.6986, f1=0.6050, recall=0.6792, precision=0.5455, score=0.6609, precisionAware=0.5940  TP:36 FP:30 FN:17 TN:45
- epoch 52: auc=0.7044, f1=0.5932, recall=0.6604, precision=0.5385, score=0.6490, precisionAware=0.5881  TP:35 FP:30 FN:18 TN:45
- epoch 58: auc=0.7175, f1=0.5965, recall=0.6415, precision=0.5574, score=0.6432, precisionAware=0.6011  TP:34 FP:27 FN:19 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 4: auc=0.6023, f1=0.1071, recall=0.0566, precision=1.0000, precisionAware=0.6526, composite=0.1809  TP:3 FP:0 FN:50 TN:75
- epoch 57: auc=0.7263, f1=0.6154, recall=0.6038, precision=0.6275, precisionAware=0.6436, composite=0.6318  TP:32 FP:19 FN:21 TN:56
- epoch 56: auc=0.7278, f1=0.5859, recall=0.5472, precision=0.6304, precisionAware=0.6365, composite=0.5949  TP:29 FP:17 FN:24 TN:58
- epoch 50: auc=0.7182, f1=0.5859, recall=0.5472, precision=0.6304, precisionAware=0.6346, composite=0.5930  TP:29 FP:17 FN:24 TN:58
