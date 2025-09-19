# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed12 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 12 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7304
- F1: 0.6852
- Recall: 0.6685
- Precision: 0.7029
- Composite Score: 0.6859 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7031 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 123
- FP: 52
- FN: 61
- TN: 118

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7643, f1=0.6870, recall=0.8491, precision=0.5769, score=0.7835, precisionAware=0.6474  TP:45 FP:33 FN:8 TN:42
- epoch 58: auc=0.7567, f1=0.6870, recall=0.8491, precision=0.5769, score=0.7820, precisionAware=0.6459  TP:45 FP:33 FN:8 TN:42
- epoch 59: auc=0.7570, f1=0.6772, recall=0.8113, precision=0.5811, score=0.7602, precisionAware=0.6451  TP:43 FP:31 FN:10 TN:44
- epoch 54: auc=0.7525, f1=0.6829, recall=0.7925, precision=0.6000, score=0.7516, precisionAware=0.6554  TP:42 FP:28 FN:11 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7821, f1=0.6731, recall=0.6604, precision=0.6863, precisionAware=0.7015, composite=0.6885  TP:35 FP:16 FN:18 TN:59
- epoch 39: auc=0.7726, f1=0.6667, recall=0.6415, precision=0.6939, precisionAware=0.7015, composite=0.6753  TP:34 FP:15 FN:19 TN:60
- epoch 36: auc=0.7605, f1=0.6731, recall=0.6604, precision=0.6863, precisionAware=0.6972, composite=0.6842  TP:35 FP:16 FN:18 TN:59
- epoch 29: auc=0.7645, f1=0.6392, recall=0.5849, precision=0.7045, precisionAware=0.6969, composite=0.6371  TP:31 FP:13 FN:22 TN:62
