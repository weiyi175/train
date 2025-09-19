# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed23 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 23 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6859
- F1: 0.6938
- Recall: 0.7880
- Precision: 0.6197
- Composite Score: 0.7393 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6551 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 145
- FP: 89
- FN: 39
- TN: 81

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7356, f1=0.6435, recall=0.6981, precision=0.5968, score=0.6892, precisionAware=0.6386  TP:37 FP:25 FN:16 TN:50
- epoch 59: auc=0.7379, f1=0.6379, recall=0.6981, precision=0.5873, score=0.6880, precisionAware=0.6326  TP:37 FP:26 FN:16 TN:49
- epoch 60: auc=0.7296, f1=0.6325, recall=0.6981, precision=0.5781, score=0.6847, precisionAware=0.6247  TP:37 FP:27 FN:16 TN:48
- epoch 56: auc=0.7208, f1=0.6379, recall=0.6981, precision=0.5873, score=0.6846, precisionAware=0.6292  TP:37 FP:26 FN:16 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7205, f1=0.6422, recall=0.6604, precision=0.6250, precisionAware=0.6493, composite=0.6669  TP:35 FP:21 FN:18 TN:54
- epoch 58: auc=0.7356, f1=0.6435, recall=0.6981, precision=0.5968, precisionAware=0.6386, composite=0.6892  TP:37 FP:25 FN:16 TN:50
- epoch 54: auc=0.7112, f1=0.6168, recall=0.6226, precision=0.6111, precisionAware=0.6328, composite=0.6386  TP:33 FP:21 FN:20 TN:54
- epoch 59: auc=0.7379, f1=0.6379, recall=0.6981, precision=0.5873, precisionAware=0.6326, composite=0.6880  TP:37 FP:26 FN:16 TN:49
