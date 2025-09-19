# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed15 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 15 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7361
- F1: 0.6630
- Recall: 0.6467
- Precision: 0.6800
- Composite Score: 0.6695 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6861 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 119
- FP: 56
- FN: 65
- TN: 114

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7789, f1=0.6721, recall=0.7736, precision=0.5942, score=0.7442, precisionAware=0.6545  TP:41 FP:28 FN:12 TN:47
- epoch 26: auc=0.7643, f1=0.6508, recall=0.7736, precision=0.5616, score=0.7349, precisionAware=0.6289  TP:41 FP:32 FN:12 TN:43
- epoch 47: auc=0.7701, f1=0.6557, recall=0.7547, precision=0.5797, score=0.7281, precisionAware=0.6406  TP:40 FP:29 FN:13 TN:46
- epoch 52: auc=0.7819, f1=0.6610, recall=0.7358, precision=0.6000, score=0.7226, precisionAware=0.6547  TP:39 FP:26 FN:14 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7879, f1=0.6733, recall=0.6415, precision=0.7083, precisionAware=0.7137, composite=0.6803  TP:34 FP:14 FN:19 TN:61
- epoch 51: auc=0.7972, f1=0.6465, recall=0.6038, precision=0.6957, precisionAware=0.7012, composite=0.6553  TP:32 FP:14 FN:21 TN:61
- epoch 42: auc=0.7819, f1=0.6465, recall=0.6038, precision=0.6957, precisionAware=0.6981, composite=0.6522  TP:32 FP:14 FN:21 TN:61
- epoch 59: auc=0.7902, f1=0.6327, recall=0.5849, precision=0.6889, precisionAware=0.6923, composite=0.6403  TP:31 FP:14 FN:22 TN:61
