# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed13 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 13 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7041
- F1: 0.6307
- Recall: 0.6033
- Precision: 0.6607
- Composite Score: 0.6317 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6604 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 111
- FP: 57
- FN: 73
- TN: 113

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7381, f1=0.6727, recall=0.6981, precision=0.6491, score=0.6985, precisionAware=0.6740  TP:37 FP:20 FN:16 TN:55
- epoch 57: auc=0.7323, f1=0.6667, recall=0.6981, precision=0.6379, score=0.6955, precisionAware=0.6654  TP:37 FP:21 FN:16 TN:54
- epoch 58: auc=0.7303, f1=0.6667, recall=0.6981, precision=0.6379, score=0.6951, precisionAware=0.6650  TP:37 FP:21 FN:16 TN:54
- epoch 54: auc=0.7258, f1=0.6542, recall=0.6604, precision=0.6481, score=0.6716, precisionAware=0.6655  TP:35 FP:19 FN:18 TN:56

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7414, f1=0.6465, recall=0.6038, precision=0.6957, precisionAware=0.6900, composite=0.6441  TP:32 FP:14 FN:21 TN:61
- epoch 56: auc=0.7381, f1=0.6727, recall=0.6981, precision=0.6491, precisionAware=0.6740, composite=0.6985  TP:37 FP:20 FN:16 TN:55
- epoch 49: auc=0.7160, f1=0.6538, recall=0.6415, precision=0.6667, precisionAware=0.6727, composite=0.6601  TP:34 FP:17 FN:19 TN:58
- epoch 54: auc=0.7258, f1=0.6542, recall=0.6604, precision=0.6481, precisionAware=0.6655, composite=0.6716  TP:35 FP:19 FN:18 TN:56
