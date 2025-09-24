# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh035/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 3 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7462
- F1: 0.6335
- Recall: 0.5543
- Precision: 0.7391
- Composite Score: 0.6165 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7089 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 102
- FP: 36
- FN: 82
- TN: 134

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6807, f1=0.5950, recall=0.5373, precision=0.6667, score=0.5833, precisionAware=0.6480  TP:36 FP:18 FN:31 TN:43
- epoch 2: auc=0.6658, f1=0.3579, recall=0.2537, precision=0.6071, score=0.3674, precisionAware=0.5441  TP:17 FP:11 FN:50 TN:50
- epoch 1: auc=0.6369, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1274, precisionAware=0.1274  TP:0 FP:0 FN:67 TN:61

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6807, f1=0.5950, recall=0.5373, precision=0.6667, precisionAware=0.6480, composite=0.5833  TP:36 FP:18 FN:31 TN:43
- epoch 2: auc=0.6658, f1=0.3579, recall=0.2537, precision=0.6071, precisionAware=0.5441, composite=0.3674  TP:17 FP:11 FN:50 TN:50
- epoch 1: auc=0.6369, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1274, composite=0.1274  TP:0 FP:0 FN:67 TN:61
