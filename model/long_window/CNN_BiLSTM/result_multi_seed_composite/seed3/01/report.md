# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed3 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 3 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6990
- F1: 0.7352
- Recall: 0.8750
- Precision: 0.6339
- Composite Score: 0.7979 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6773 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 161
- FP: 93
- FN: 23
- TN: 77

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 67: auc=0.7971, f1=0.7568, recall=0.8889, precision=0.6588, score=0.8309, precisionAware=0.7159  TP:56 FP:29 FN:7 TN:36
- epoch 70: auc=0.7834, f1=0.7571, recall=0.8413, precision=0.6883, score=0.8045, precisionAware=0.7280  TP:53 FP:24 FN:10 TN:41
- epoch 58: auc=0.7597, f1=0.7194, recall=0.7937, precision=0.6579, score=0.7646, precisionAware=0.6967  TP:50 FP:26 FN:13 TN:39
- epoch 54: auc=0.7609, f1=0.7111, recall=0.7619, precision=0.6667, score=0.7465, precisionAware=0.6989  TP:48 FP:24 FN:15 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7714, f1=0.6949, recall=0.6508, precision=0.7455, precisionAware=0.7355, composite=0.6882  TP:41 FP:14 FN:22 TN:51
- epoch 13: auc=0.7346, f1=0.5773, recall=0.4444, precision=0.8235, precisionAware=0.7319, composite=0.5423  TP:28 FP:6 FN:35 TN:59
- epoch 70: auc=0.7834, f1=0.7571, recall=0.8413, precision=0.6883, precisionAware=0.7280, composite=0.8045  TP:53 FP:24 FN:10 TN:41
- epoch 14: auc=0.7336, f1=0.6154, recall=0.5079, precision=0.7805, precisionAware=0.7216, composite=0.5853  TP:32 FP:9 FN:31 TN:56
