# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed18 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 18 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6919
- F1: 0.6507
- Recall: 0.6630
- Precision: 0.6387
- Composite Score: 0.6651 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6530 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 69
- FN: 62
- TN: 101

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7663, f1=0.6610, recall=0.7358, precision=0.6000, score=0.7195, precisionAware=0.6516  TP:39 FP:26 FN:14 TN:49
- epoch 56: auc=0.7618, f1=0.6610, recall=0.7358, precision=0.6000, score=0.7186, precisionAware=0.6507  TP:39 FP:26 FN:14 TN:49
- epoch 54: auc=0.7592, f1=0.6555, recall=0.7358, precision=0.5909, score=0.7164, precisionAware=0.6439  TP:39 FP:27 FN:14 TN:48
- epoch 57: auc=0.7618, f1=0.6667, recall=0.7170, precision=0.6230, score=0.7108, precisionAware=0.6638  TP:38 FP:23 FN:15 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7618, f1=0.6667, recall=0.7170, precision=0.6230, precisionAware=0.6638, composite=0.7108  TP:38 FP:23 FN:15 TN:52
- epoch 59: auc=0.7610, f1=0.6422, recall=0.6604, precision=0.6250, precisionAware=0.6574, composite=0.6751  TP:35 FP:21 FN:18 TN:54
- epoch 60: auc=0.7663, f1=0.6610, recall=0.7358, precision=0.6000, precisionAware=0.6516, composite=0.7195  TP:39 FP:26 FN:14 TN:49
- epoch 56: auc=0.7618, f1=0.6610, recall=0.7358, precision=0.6000, precisionAware=0.6507, composite=0.7186  TP:39 FP:26 FN:14 TN:49
