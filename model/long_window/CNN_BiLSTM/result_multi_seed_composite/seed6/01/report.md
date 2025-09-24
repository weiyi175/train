# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed6 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 6 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6920
- F1: 0.6310
- Recall: 0.5761
- Precision: 0.6974
- Composite Score: 0.6157 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6764 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 106
- FP: 46
- FN: 78
- TN: 124

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7448, f1=0.7361, recall=0.8413, precision=0.6543, score=0.7904, precisionAware=0.6970  TP:53 FP:28 FN:10 TN:37
- epoch 47: auc=0.7639, f1=0.7194, recall=0.7937, precision=0.6579, score=0.7654, precisionAware=0.6975  TP:50 FP:26 FN:13 TN:39
- epoch 48: auc=0.7585, f1=0.7101, recall=0.7778, precision=0.6533, score=0.7536, precisionAware=0.6914  TP:49 FP:26 FN:14 TN:39
- epoch 66: auc=0.7683, f1=0.7007, recall=0.7619, precision=0.6486, score=0.7448, precisionAware=0.6882  TP:48 FP:26 FN:15 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.8054, f1=0.6186, recall=0.4762, precision=0.8824, precisionAware=0.7878, composite=0.5847  TP:30 FP:4 FN:33 TN:61
- epoch 14: auc=0.7617, f1=0.6186, recall=0.4762, precision=0.8824, precisionAware=0.7791, composite=0.5760  TP:30 FP:4 FN:33 TN:61
- epoch 13: auc=0.7541, f1=0.6042, recall=0.4603, precision=0.8788, precisionAware=0.7715, composite=0.5622  TP:29 FP:4 FN:34 TN:61
- epoch 15: auc=0.7607, f1=0.5652, recall=0.4127, precision=0.8966, precisionAware=0.7700, composite=0.5281  TP:26 FP:3 FN:37 TN:62
