# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed2 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 2 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7476
- F1: 0.7078
- Recall: 0.7174
- Precision: 0.6984
- Composite Score: 0.7205 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7111 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 132
- FP: 57
- FN: 52
- TN: 113

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7635, f1=0.6842, recall=0.7358, precision=0.6393, score=0.7259, precisionAware=0.6776  TP:39 FP:22 FN:14 TN:53
- epoch 51: auc=0.7600, f1=0.6724, recall=0.7358, precision=0.6190, score=0.7216, precisionAware=0.6632  TP:39 FP:24 FN:14 TN:51
- epoch 50: auc=0.7464, f1=0.6667, recall=0.7358, precision=0.6094, score=0.7172, precisionAware=0.6540  TP:39 FP:25 FN:14 TN:50
- epoch 61: auc=0.7396, f1=0.6610, recall=0.7358, precision=0.6000, score=0.7142, precisionAware=0.6462  TP:39 FP:26 FN:14 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7635, f1=0.6842, recall=0.7358, precision=0.6393, precisionAware=0.6776, composite=0.7259  TP:39 FP:22 FN:14 TN:53
- epoch 33: auc=0.7580, f1=0.6607, recall=0.6981, precision=0.6271, precisionAware=0.6634, composite=0.6989  TP:37 FP:22 FN:16 TN:53
- epoch 51: auc=0.7600, f1=0.6724, recall=0.7358, precision=0.6190, precisionAware=0.6632, composite=0.7216  TP:39 FP:24 FN:14 TN:51
- epoch 60: auc=0.7801, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6619, composite=0.6559  TP:33 FP:19 FN:20 TN:56
