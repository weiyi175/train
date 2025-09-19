# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed9 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 9 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7091
- F1: 0.6578
- Recall: 0.6685
- Precision: 0.6474
- Composite Score: 0.6734 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6628 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 123
- FP: 67
- FN: 61
- TN: 103

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7537, f1=0.6667, recall=0.7736, precision=0.5857, score=0.7375, precisionAware=0.6436  TP:41 FP:29 FN:12 TN:46
- epoch 44: auc=0.7293, f1=0.6613, recall=0.7736, precision=0.5775, score=0.7310, precisionAware=0.6330  TP:41 FP:30 FN:12 TN:45
- epoch 46: auc=0.7459, f1=0.6723, recall=0.7547, precision=0.6061, score=0.7282, precisionAware=0.6539  TP:40 FP:26 FN:13 TN:49
- epoch 50: auc=0.7464, f1=0.6612, recall=0.7547, precision=0.5882, score=0.7250, precisionAware=0.6417  TP:40 FP:28 FN:13 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 38: auc=0.7592, f1=0.6105, recall=0.5472, precision=0.6905, precisionAware=0.6802, composite=0.6086  TP:29 FP:13 FN:24 TN:62
- epoch 45: auc=0.7655, f1=0.6042, recall=0.5472, precision=0.6744, precisionAware=0.6716, composite=0.6079  TP:29 FP:14 FN:24 TN:61
- epoch 29: auc=0.7560, f1=0.5806, recall=0.5094, precision=0.6750, precisionAware=0.6629, composite=0.5801  TP:27 FP:13 FN:26 TN:62
- epoch 46: auc=0.7459, f1=0.6723, recall=0.7547, precision=0.6061, precisionAware=0.6539, composite=0.7282  TP:40 FP:26 FN:13 TN:49
