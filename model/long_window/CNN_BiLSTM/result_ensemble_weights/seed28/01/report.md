# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed28 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 28 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7199
- F1: 0.7103
- Recall: 0.8261
- Precision: 0.6230
- Composite Score: 0.7701 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6685 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 152
- FP: 92
- FN: 32
- TN: 78

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7200, f1=0.6466, recall=0.8113, precision=0.5375, score=0.7436, precisionAware=0.6067  TP:43 FP:37 FN:10 TN:38
- epoch 59: auc=0.7286, f1=0.6462, recall=0.7925, precision=0.5455, score=0.7358, precisionAware=0.6123  TP:42 FP:35 FN:11 TN:40
- epoch 58: auc=0.7401, f1=0.6441, recall=0.7170, precision=0.5846, score=0.6997, precisionAware=0.6336  TP:38 FP:27 FN:15 TN:48
- epoch 54: auc=0.7240, f1=0.6271, recall=0.6981, precision=0.5692, score=0.6820, precisionAware=0.6176  TP:37 FP:28 FN:16 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7401, f1=0.6441, recall=0.7170, precision=0.5846, precisionAware=0.6336, composite=0.6997  TP:38 FP:27 FN:15 TN:48
- epoch 54: auc=0.7240, f1=0.6271, recall=0.6981, precision=0.5692, precisionAware=0.6176, composite=0.6820  TP:37 FP:28 FN:16 TN:47
- epoch 50: auc=0.7195, f1=0.6071, recall=0.6415, precision=0.5763, precisionAware=0.6142, composite=0.6468  TP:34 FP:25 FN:19 TN:50
- epoch 59: auc=0.7286, f1=0.6462, recall=0.7925, precision=0.5455, precisionAware=0.6123, composite=0.7358  TP:42 FP:35 FN:11 TN:40
