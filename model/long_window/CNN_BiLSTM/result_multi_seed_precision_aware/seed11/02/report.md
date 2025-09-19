# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed11 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 11 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6653
- F1: 0.6683
- Recall: 0.7554
- Precision: 0.5991
- Composite Score: 0.7113 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6331 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 139
- FP: 93
- FN: 45
- TN: 77

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7102, f1=0.6341, recall=0.7358, precision=0.5571, score=0.7002, precisionAware=0.6109  TP:39 FP:31 FN:14 TN:44
- epoch 48: auc=0.6958, f1=0.6393, recall=0.7358, precision=0.5652, score=0.6989, precisionAware=0.6136  TP:39 FP:30 FN:14 TN:45
- epoch 46: auc=0.6916, f1=0.6000, recall=0.7358, precision=0.5065, score=0.6862, precisionAware=0.5716  TP:39 FP:38 FN:14 TN:37
- epoch 51: auc=0.7092, f1=0.6102, recall=0.6792, precision=0.5538, score=0.6645, precisionAware=0.6018  TP:36 FP:29 FN:17 TN:46

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7560, f1=0.6139, recall=0.5849, precision=0.6458, precisionAware=0.6583, composite=0.6278  TP:31 FP:17 FN:22 TN:58
- epoch 58: auc=0.7472, f1=0.6095, recall=0.6038, precision=0.6154, precisionAware=0.6400, composite=0.6342  TP:32 FP:20 FN:21 TN:55
- epoch 23: auc=0.7127, f1=0.5000, recall=0.3962, precision=0.6774, precisionAware=0.6313, composite=0.4907  TP:21 FP:10 FN:32 TN:65
- epoch 20: auc=0.6790, f1=0.5773, recall=0.5283, precision=0.6364, precisionAware=0.6272, composite=0.5731  TP:28 FP:16 FN:25 TN:59
