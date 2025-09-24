# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed1 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 1 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7014
- F1: 0.6269
- Recall: 0.5707
- Precision: 0.6954
- Composite Score: 0.6137 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6760 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 105
- FP: 46
- FN: 79
- TN: 124

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 69: auc=0.8093, f1=0.7119, recall=0.7778, precision=0.6562, score=0.7643, precisionAware=0.7035  TP:42 FP:22 FN:12 TN:52
- epoch 60: auc=0.8023, f1=0.7119, recall=0.7778, precision=0.6562, score=0.7629, precisionAware=0.7021  TP:42 FP:22 FN:12 TN:52
- epoch 68: auc=0.7873, f1=0.7179, recall=0.7778, precision=0.6667, score=0.7617, precisionAware=0.7062  TP:42 FP:21 FN:12 TN:53
- epoch 63: auc=0.7938, f1=0.7119, recall=0.7778, precision=0.6562, score=0.7612, precisionAware=0.7004  TP:42 FP:22 FN:12 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 67: auc=0.8161, f1=0.6972, recall=0.7037, precision=0.6909, precisionAware=0.7178, composite=0.7242  TP:38 FP:17 FN:16 TN:57
- epoch 42: auc=0.7938, f1=0.6667, recall=0.6296, precision=0.7083, precisionAware=0.7129, composite=0.6736  TP:34 FP:14 FN:20 TN:60
- epoch 65: auc=0.7953, f1=0.7080, recall=0.7407, precision=0.6780, precisionAware=0.7104, composite=0.7418  TP:40 FP:19 FN:14 TN:55
- epoch 68: auc=0.7873, f1=0.7179, recall=0.7778, precision=0.6667, precisionAware=0.7062, composite=0.7617  TP:42 FP:21 FN:12 TN:53
