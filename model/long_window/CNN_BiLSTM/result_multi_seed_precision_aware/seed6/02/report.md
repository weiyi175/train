# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed6 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 6 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7315
- F1: 0.5185
- Recall: 0.3804
- Precision: 0.8140
- Composite Score: 0.4921 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7088 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 70
- FP: 16
- FN: 114
- TN: 154

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7099, f1=0.6218, recall=0.6981, precision=0.5606, score=0.6776, precisionAware=0.6088  TP:37 FP:29 FN:16 TN:46
- epoch 59: auc=0.6911, f1=0.6218, recall=0.6981, precision=0.5606, score=0.6738, precisionAware=0.6051  TP:37 FP:29 FN:16 TN:46
- epoch 54: auc=0.7303, f1=0.6261, recall=0.6792, precision=0.5806, score=0.6735, precisionAware=0.6242  TP:36 FP:26 FN:17 TN:49
- epoch 41: auc=0.7311, f1=0.6306, recall=0.6604, precision=0.6034, score=0.6656, precisionAware=0.6371  TP:35 FP:23 FN:18 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 36: auc=0.7630, f1=0.5238, recall=0.4151, precision=0.7097, precisionAware=0.6646, composite=0.5173  TP:22 FP:9 FN:31 TN:66
- epoch 45: auc=0.7449, f1=0.5745, recall=0.5094, precision=0.6585, precisionAware=0.6506, composite=0.5760  TP:27 FP:14 FN:26 TN:61
- epoch 28: auc=0.7240, f1=0.5556, recall=0.4717, precision=0.6757, precisionAware=0.6493, composite=0.5473  TP:25 FP:12 FN:28 TN:63
- epoch 55: auc=0.7673, f1=0.5773, recall=0.5283, precision=0.6364, precisionAware=0.6448, composite=0.5908  TP:28 FP:16 FN:25 TN:59
