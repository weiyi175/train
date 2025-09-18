# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed12 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 12 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7090
- F1: 0.5641
- Recall: 0.5238
- Precision: 0.6111
- Composite Score: 0.5729 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6166 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 35
- FN: 50
- TN: 116

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7082, f1=0.6087, recall=0.6604, precision=0.5645, score=0.6544, precisionAware=0.6065  TP:35 FP:27 FN:18 TN:48
- epoch 59: auc=0.7399, f1=0.6126, recall=0.6415, precision=0.5862, score=0.6525, precisionAware=0.6249  TP:34 FP:24 FN:19 TN:51
- epoch 52: auc=0.7333, f1=0.6226, recall=0.6226, precision=0.6226, score=0.6448, precisionAware=0.6448  TP:33 FP:20 FN:20 TN:55
- epoch 46: auc=0.7104, f1=0.6168, recall=0.6226, precision=0.6111, score=0.6385, precisionAware=0.6327  TP:33 FP:21 FN:20 TN:54

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7623, f1=0.6250, recall=0.5660, precision=0.6977, precisionAware=0.6888, composite=0.6230  TP:30 FP:13 FN:23 TN:62
- epoch 60: auc=0.7565, f1=0.5870, recall=0.5094, precision=0.6923, precisionAware=0.6735, composite=0.5821  TP:27 FP:12 FN:26 TN:63
- epoch 47: auc=0.7434, f1=0.6042, recall=0.5472, precision=0.6744, precisionAware=0.6671, composite=0.6035  TP:29 FP:14 FN:24 TN:61
- epoch 35: auc=0.7275, f1=0.5979, recall=0.5472, precision=0.6591, precisionAware=0.6544, composite=0.5985  TP:29 FP:15 FN:24 TN:60
