# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed5 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 5 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7691
- F1: 0.6291
- Recall: 0.6381
- Precision: 0.6204
- Composite Score: 0.6616 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6527 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 67
- FP: 41
- FN: 38
- TN: 110

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.7733, f1=0.6774, recall=0.7925, precision=0.5915, score=0.7541, precisionAware=0.6537  TP:42 FP:29 FN:11 TN:46
- epoch 52: auc=0.7532, f1=0.6667, recall=0.7925, precision=0.5753, score=0.7469, precisionAware=0.6383  TP:42 FP:31 FN:11 TN:44
- epoch 59: auc=0.7872, f1=0.6909, recall=0.7170, precision=0.6667, score=0.7232, precisionAware=0.6980  TP:38 FP:19 FN:15 TN:56
- epoch 50: auc=0.7847, f1=0.6847, recall=0.7170, precision=0.6552, score=0.7208, precisionAware=0.6899  TP:38 FP:20 FN:15 TN:55

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7947, f1=0.6667, recall=0.6038, precision=0.7442, precisionAware=0.7310, composite=0.6608  TP:32 FP:11 FN:21 TN:64
- epoch 47: auc=0.7987, f1=0.6598, recall=0.6038, precision=0.7273, precisionAware=0.7213, composite=0.6596  TP:32 FP:12 FN:21 TN:63
- epoch 38: auc=0.7811, f1=0.6733, recall=0.6415, precision=0.7083, precisionAware=0.7124, composite=0.6790  TP:34 FP:14 FN:19 TN:61
- epoch 59: auc=0.7872, f1=0.6909, recall=0.7170, precision=0.6667, precisionAware=0.6980, composite=0.7232  TP:38 FP:19 FN:15 TN:56
