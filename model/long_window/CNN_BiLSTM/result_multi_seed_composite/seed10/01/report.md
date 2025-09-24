# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed10 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 10 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7141
- F1: 0.6897
- Recall: 0.7065
- Precision: 0.6736
- Composite Score: 0.7030 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6865 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 130
- FP: 63
- FN: 54
- TN: 107

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 44: auc=0.7729, f1=0.7077, recall=0.7302, precision=0.6866, score=0.7320, precisionAware=0.7102  TP:46 FP:21 FN:17 TN:44
- epoch 54: auc=0.7377, f1=0.7077, recall=0.7302, precision=0.6866, score=0.7249, precisionAware=0.7031  TP:46 FP:21 FN:17 TN:44
- epoch 58: auc=0.7619, f1=0.6825, recall=0.6825, precision=0.6825, score=0.6984, precisionAware=0.6984  TP:43 FP:20 FN:20 TN:45
- epoch 59: auc=0.7595, f1=0.6825, recall=0.6825, precision=0.6825, score=0.6979, precisionAware=0.6979  TP:43 FP:20 FN:20 TN:45

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 13: auc=0.7280, f1=0.6154, recall=0.5079, precision=0.7805, precisionAware=0.7205, composite=0.5842  TP:32 FP:9 FN:31 TN:56
- epoch 52: auc=0.7631, f1=0.6429, recall=0.5714, precision=0.7347, precisionAware=0.7128, composite=0.6312  TP:36 FP:13 FN:27 TN:52
- epoch 57: auc=0.7424, f1=0.6095, recall=0.5079, precision=0.7619, precisionAware=0.7123, composite=0.5853  TP:32 FP:10 FN:31 TN:55
- epoch 44: auc=0.7729, f1=0.7077, recall=0.7302, precision=0.6866, precisionAware=0.7102, composite=0.7320  TP:46 FP:21 FN:17 TN:44
