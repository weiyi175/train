# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7533
- F1: 0.5856
- Recall: 0.5048
- Precision: 0.6974
- Composite Score: 0.5787 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6750 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 53
- FP: 23
- FN: 52
- TN: 128

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 61: auc=0.7907, f1=0.6852, recall=0.6981, precision=0.6727, score=0.7128, precisionAware=0.7001  TP:37 FP:18 FN:16 TN:57
- epoch 48: auc=0.7746, f1=0.6606, recall=0.6792, precision=0.6429, score=0.6927, precisionAware=0.6745  TP:36 FP:20 FN:17 TN:55
- epoch 59: auc=0.7801, f1=0.6486, recall=0.6792, precision=0.6207, score=0.6902, precisionAware=0.6610  TP:36 FP:22 FN:17 TN:53
- epoch 55: auc=0.7847, f1=0.6667, recall=0.6604, precision=0.6731, score=0.6871, precisionAware=0.6935  TP:35 FP:17 FN:18 TN:58

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7940, f1=0.6667, recall=0.6415, precision=0.6939, precisionAware=0.7057, composite=0.6795  TP:34 FP:15 FN:19 TN:60
- epoch 61: auc=0.7907, f1=0.6852, recall=0.6981, precision=0.6727, precisionAware=0.7001, composite=0.7128  TP:37 FP:18 FN:16 TN:57
- epoch 55: auc=0.7847, f1=0.6667, recall=0.6604, precision=0.6731, precisionAware=0.6935, composite=0.6871  TP:35 FP:17 FN:18 TN:58
- epoch 47: auc=0.7753, f1=0.6471, recall=0.6226, precision=0.6735, precisionAware=0.6859, composite=0.6605  TP:33 FP:16 FN:20 TN:59
