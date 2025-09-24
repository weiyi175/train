# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 50 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6957
- F1: 0.7040
- Recall: 0.8207
- Precision: 0.6163
- Composite Score: 0.7607 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6585 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 151
- FP: 94
- FN: 33
- TN: 76

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 42: auc=0.7426, f1=0.7639, recall=0.8730, precision=0.6790, score=0.8142, precisionAware=0.7172  TP:55 FP:26 FN:8 TN:39
- epoch 48: auc=0.7580, f1=0.7162, recall=0.8413, precision=0.6235, score=0.7871, precisionAware=0.6782  TP:53 FP:32 FN:10 TN:33
- epoch 30: auc=0.7602, f1=0.7222, recall=0.8254, precision=0.6420, score=0.7814, precisionAware=0.6897  TP:52 FP:29 FN:11 TN:36
- epoch 45: auc=0.7636, f1=0.7299, recall=0.7937, precision=0.6757, score=0.7685, precisionAware=0.7095  TP:50 FP:24 FN:13 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 14: auc=0.7377, f1=0.7000, recall=0.6667, precision=0.7368, precisionAware=0.7260, composite=0.6909  TP:42 FP:15 FN:21 TN:50
- epoch 12: auc=0.7270, f1=0.6355, recall=0.5397, precision=0.7727, precisionAware=0.7224, composite=0.6059  TP:34 FP:10 FN:29 TN:55
- epoch 37: auc=0.7560, f1=0.7231, recall=0.7460, precision=0.7015, precisionAware=0.7189, composite=0.7411  TP:47 FP:20 FN:16 TN:45
- epoch 22: auc=0.7519, f1=0.7231, recall=0.7460, precision=0.7015, precisionAware=0.7180, composite=0.7403  TP:47 FP:20 FN:16 TN:45
