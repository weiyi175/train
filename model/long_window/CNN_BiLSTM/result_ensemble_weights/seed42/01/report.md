# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed42 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7034
- F1: 0.6860
- Recall: 0.7717
- Precision: 0.6174
- Composite Score: 0.7324 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6552 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 142
- FP: 88
- FN: 42
- TN: 82

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7067, f1=0.6457, recall=0.7736, precision=0.5541, score=0.7218, precisionAware=0.6121  TP:41 FP:33 FN:12 TN:42
- epoch 55: auc=0.6933, f1=0.6308, recall=0.7736, precision=0.5325, score=0.7147, precisionAware=0.5941  TP:41 FP:36 FN:12 TN:39
- epoch 59: auc=0.7119, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7117, precisionAware=0.6122  TP:40 FP:32 FN:13 TN:43
- epoch 50: auc=0.7001, f1=0.6202, recall=0.7547, precision=0.5263, score=0.7034, precisionAware=0.5892  TP:40 FP:36 FN:13 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7225, f1=0.6491, recall=0.6981, precision=0.6066, precisionAware=0.6425, composite=0.6883  TP:37 FP:24 FN:16 TN:51
- epoch 57: auc=0.7182, f1=0.6435, recall=0.6981, precision=0.5968, precisionAware=0.6351, composite=0.6857  TP:37 FP:25 FN:16 TN:50
- epoch 52: auc=0.7195, f1=0.6306, recall=0.6604, precision=0.6034, precisionAware=0.6348, composite=0.6633  TP:35 FP:23 FN:18 TN:52
- epoch 53: auc=0.7213, f1=0.6182, recall=0.6415, precision=0.5965, precisionAware=0.6280, composite=0.6505  TP:34 FP:23 FN:19 TN:52
