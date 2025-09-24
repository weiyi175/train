# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_strideeqwin/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 50 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7183
- F1: 0.4059
- Recall: 0.2989
- Precision: 0.6322
- Composite Score: 0.4149 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5815 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 32
- FN: 129
- TN: 138

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6014, f1=0.5301, recall=0.6875, precision=0.4314, score=0.6231, precisionAware=0.4950  TP:22 FP:29 FN:10 TN:12
- epoch 30: auc=0.6936, f1=0.6230, recall=0.5938, precision=0.6552, score=0.6225, precisionAware=0.6532  TP:19 FP:10 FN:13 TN:31
- epoch 28: auc=0.6936, f1=0.5714, recall=0.5000, precision=0.6667, score=0.5601, precisionAware=0.6435  TP:16 FP:8 FN:16 TN:33
- epoch 29: auc=0.6898, f1=0.5714, recall=0.5000, precision=0.6667, score=0.5594, precisionAware=0.6427  TP:16 FP:8 FN:16 TN:33

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 7: auc=0.6951, f1=0.1176, recall=0.0625, precision=1.0000, precisionAware=0.6743, composite=0.2056  TP:2 FP:0 FN:30 TN:41
- epoch 6: auc=0.6966, f1=0.0606, recall=0.0312, precision=1.0000, precisionAware=0.6575, composite=0.1731  TP:1 FP:0 FN:31 TN:41
- epoch 30: auc=0.6936, f1=0.6230, recall=0.5938, precision=0.6552, precisionAware=0.6532, composite=0.6225  TP:19 FP:10 FN:13 TN:31
- epoch 28: auc=0.6936, f1=0.5714, recall=0.5000, precision=0.6667, precisionAware=0.6435, composite=0.5601  TP:16 FP:8 FN:16 TN:33
