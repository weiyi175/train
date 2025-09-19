# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed18 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 18 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6807
- F1: 0.6847
- Recall: 0.7554
- Precision: 0.6261
- Composite Score: 0.7193 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6546 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 139
- FP: 83
- FN: 45
- TN: 87

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 45: auc=0.6830, f1=0.5970, recall=0.7547, precision=0.4938, score=0.6931, precisionAware=0.5626  TP:40 FP:41 FN:13 TN:34
- epoch 47: auc=0.6886, f1=0.5954, recall=0.7358, precision=0.5000, score=0.6843, precisionAware=0.5663  TP:39 FP:39 FN:14 TN:36
- epoch 60: auc=0.7077, f1=0.6080, recall=0.7170, precision=0.5278, score=0.6824, precisionAware=0.5878  TP:38 FP:34 FN:15 TN:41
- epoch 58: auc=0.7029, f1=0.6116, recall=0.6981, precision=0.5441, score=0.6731, precisionAware=0.5961  TP:37 FP:31 FN:16 TN:44

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 37: auc=0.7414, f1=0.5435, recall=0.4717, precision=0.6410, precisionAware=0.6318, composite=0.5472  TP:25 FP:14 FN:28 TN:61
- epoch 24: auc=0.6948, f1=0.5859, recall=0.5472, precision=0.6304, precisionAware=0.6299, composite=0.5883  TP:29 FP:17 FN:24 TN:58
- epoch 30: auc=0.7195, f1=0.5714, recall=0.5283, precision=0.6222, precisionAware=0.6264, composite=0.5795  TP:28 FP:17 FN:25 TN:58
- epoch 48: auc=0.7364, f1=0.5417, recall=0.4906, precision=0.6047, precisionAware=0.6121, composite=0.5551  TP:26 FP:17 FN:27 TN:58
