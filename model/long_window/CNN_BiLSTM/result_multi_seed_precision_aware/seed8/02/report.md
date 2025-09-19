# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed8 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 8 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7317
- F1: 0.6925
- Recall: 0.7283
- Precision: 0.6601
- Composite Score: 0.7182 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6841 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 134
- FP: 69
- FN: 50
- TN: 101

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7268, f1=0.6512, recall=0.7925, precision=0.5526, score=0.7369, precisionAware=0.6170  TP:42 FP:34 FN:11 TN:41
- epoch 55: auc=0.7434, f1=0.6379, recall=0.6981, precision=0.5873, score=0.6891, precisionAware=0.6337  TP:37 FP:26 FN:16 TN:49
- epoch 60: auc=0.7477, f1=0.6325, recall=0.6981, precision=0.5781, score=0.6883, precisionAware=0.6283  TP:37 FP:27 FN:16 TN:48
- epoch 39: auc=0.7512, f1=0.6606, recall=0.6792, precision=0.6429, score=0.6880, precisionAware=0.6698  TP:36 FP:20 FN:17 TN:55

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 25: auc=0.7336, f1=0.6263, recall=0.5849, precision=0.6739, precisionAware=0.6716, composite=0.6270  TP:31 FP:15 FN:22 TN:60
- epoch 31: auc=0.7535, f1=0.5957, recall=0.5283, precision=0.6829, precisionAware=0.6709, composite=0.5936  TP:28 FP:13 FN:25 TN:62
- epoch 26: auc=0.7431, f1=0.6476, recall=0.6415, precision=0.6538, precisionAware=0.6698, composite=0.6637  TP:34 FP:18 FN:19 TN:57
- epoch 39: auc=0.7512, f1=0.6606, recall=0.6792, precision=0.6429, precisionAware=0.6698, composite=0.6880  TP:36 FP:20 FN:17 TN:55
