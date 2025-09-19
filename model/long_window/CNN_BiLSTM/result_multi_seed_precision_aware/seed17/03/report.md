# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed17 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 17 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6999
- F1: 0.7012
- Recall: 0.7717
- Precision: 0.6425
- Composite Score: 0.7362 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6716 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 142
- FP: 79
- FN: 42
- TN: 91

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7414, f1=0.6542, recall=0.6604, precision=0.6481, score=0.6747, precisionAware=0.6686  TP:35 FP:19 FN:18 TN:56
- epoch 60: auc=0.7484, f1=0.5902, recall=0.6792, precision=0.5217, score=0.6664, precisionAware=0.5876  TP:36 FP:33 FN:17 TN:42
- epoch 37: auc=0.7218, f1=0.5946, recall=0.6226, precision=0.5690, score=0.6341, precisionAware=0.6072  TP:33 FP:25 FN:20 TN:50
- epoch 56: auc=0.7258, f1=0.6095, recall=0.6038, precision=0.6154, score=0.6299, precisionAware=0.6357  TP:32 FP:20 FN:21 TN:55

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7713, f1=0.6170, recall=0.5472, precision=0.7073, precisionAware=0.6930, composite=0.6130  TP:29 FP:12 FN:24 TN:63
- epoch 54: auc=0.7852, f1=0.6105, recall=0.5472, precision=0.6905, precisionAware=0.6854, composite=0.6138  TP:29 FP:13 FN:24 TN:62
- epoch 48: auc=0.7610, f1=0.6022, recall=0.5283, precision=0.7000, precisionAware=0.6828, composite=0.5970  TP:28 FP:12 FN:25 TN:63
- epoch 58: auc=0.7625, f1=0.5778, recall=0.4906, precision=0.7027, precisionAware=0.6772, composite=0.5711  TP:26 FP:11 FN:27 TN:64
