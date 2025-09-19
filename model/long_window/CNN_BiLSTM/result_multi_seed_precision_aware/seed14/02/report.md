# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed14 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 14 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7402
- F1: 0.6961
- Recall: 0.7717
- Precision: 0.6339
- Composite Score: 0.7427 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6738 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 142
- FP: 82
- FN: 42
- TN: 88

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.7535, f1=0.6667, recall=0.8113, precision=0.5658, score=0.7564, precisionAware=0.6336  TP:43 FP:33 FN:10 TN:42
- epoch 25: auc=0.7482, f1=0.6667, recall=0.7925, precision=0.5753, score=0.7459, precisionAware=0.6373  TP:42 FP:31 FN:11 TN:44
- epoch 60: auc=0.7615, f1=0.6557, recall=0.7547, precision=0.5797, score=0.7264, precisionAware=0.6389  TP:40 FP:29 FN:13 TN:46
- epoch 55: auc=0.7897, f1=0.6549, recall=0.6981, precision=0.6167, score=0.7035, precisionAware=0.6627  TP:37 FP:23 FN:16 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 46: auc=0.7887, f1=0.5909, recall=0.4906, precision=0.7429, precisionAware=0.7064, composite=0.5803  TP:26 FP:9 FN:27 TN:66
- epoch 30: auc=0.7824, f1=0.6087, recall=0.5283, precision=0.7179, precisionAware=0.6981, composite=0.6032  TP:28 FP:11 FN:25 TN:64
- epoch 58: auc=0.8078, f1=0.5843, recall=0.4906, precision=0.7222, precisionAware=0.6980, composite=0.5821  TP:26 FP:10 FN:27 TN:65
- epoch 56: auc=0.8060, f1=0.6400, recall=0.6038, precision=0.6809, precisionAware=0.6936, composite=0.6551  TP:32 FP:15 FN:21 TN:60
