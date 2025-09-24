# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed7 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 7 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7118
- F1: 0.6993
- Recall: 0.7772
- Precision: 0.6356
- Composite Score: 0.7407 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6699 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 143
- FP: 82
- FN: 41
- TN: 88

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 68: auc=0.7748, f1=0.7500, recall=0.8571, precision=0.6667, score=0.8085, precisionAware=0.7133  TP:54 FP:27 FN:9 TN:38
- epoch 65: auc=0.7810, f1=0.7424, recall=0.7778, precision=0.7101, score=0.7678, precisionAware=0.7340  TP:49 FP:20 FN:14 TN:45
- epoch 66: auc=0.7819, f1=0.7313, recall=0.7778, precision=0.6901, score=0.7647, precisionAware=0.7209  TP:49 FP:22 FN:14 TN:43
- epoch 46: auc=0.7702, f1=0.7206, recall=0.7778, precision=0.6712, score=0.7591, precisionAware=0.7058  TP:49 FP:24 FN:14 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7868, f1=0.7419, recall=0.7302, precision=0.7541, precisionAware=0.7570, composite=0.7450  TP:46 FP:15 FN:17 TN:50
- epoch 16: auc=0.7707, f1=0.7018, recall=0.6349, precision=0.7843, precisionAware=0.7568, composite=0.6821  TP:40 FP:11 FN:23 TN:54
- epoch 61: auc=0.7785, f1=0.7213, recall=0.6984, precision=0.7458, precisionAware=0.7450, composite=0.7213  TP:44 FP:15 FN:19 TN:50
- epoch 55: auc=0.7814, f1=0.6842, recall=0.6190, precision=0.7647, precisionAware=0.7439, composite=0.6711  TP:39 FP:12 FN:24 TN:53
