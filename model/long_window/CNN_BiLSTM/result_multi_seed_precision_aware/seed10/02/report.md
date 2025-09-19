# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed10 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 10 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7247
- F1: 0.6250
- Recall: 0.5707
- Precision: 0.6908
- Composite Score: 0.6178 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6778 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 105
- FP: 47
- FN: 79
- TN: 123

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.6974, f1=0.6116, recall=0.6981, precision=0.5441, score=0.6720, precisionAware=0.5950  TP:37 FP:31 FN:16 TN:44
- epoch 44: auc=0.7457, f1=0.6415, recall=0.6415, precision=0.6415, score=0.6623, precisionAware=0.6623  TP:34 FP:19 FN:19 TN:56
- epoch 54: auc=0.7346, f1=0.5983, recall=0.6604, precision=0.5469, score=0.6566, precisionAware=0.5998  TP:35 FP:29 FN:18 TN:46
- epoch 49: auc=0.7197, f1=0.6034, recall=0.6604, precision=0.5556, score=0.6552, precisionAware=0.6028  TP:35 FP:28 FN:18 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 44: auc=0.7457, f1=0.6415, recall=0.6415, precision=0.6415, precisionAware=0.6623, composite=0.6623  TP:34 FP:19 FN:19 TN:56
- epoch 50: auc=0.7761, f1=0.6139, recall=0.5849, precision=0.6458, precisionAware=0.6623, composite=0.6318  TP:31 FP:17 FN:22 TN:58
- epoch 3: auc=0.6010, f1=0.1071, recall=0.0566, precision=1.0000, precisionAware=0.6523, composite=0.1806  TP:3 FP:0 FN:50 TN:75
- epoch 51: auc=0.7494, f1=0.6226, recall=0.6226, precision=0.6226, precisionAware=0.6480, composite=0.6480  TP:33 FP:20 FN:20 TN:55
