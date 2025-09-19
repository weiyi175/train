# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed8 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 8 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6841
- F1: 0.6333
- Recall: 0.6196
- Precision: 0.6477
- Composite Score: 0.6366 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6507 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 114
- FP: 62
- FN: 70
- TN: 108

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7208, f1=0.6504, recall=0.7547, precision=0.5714, score=0.7166, precisionAware=0.6250  TP:40 FP:30 FN:13 TN:45
- epoch 57: auc=0.7200, f1=0.6393, recall=0.7358, precision=0.5652, score=0.7037, precisionAware=0.6184  TP:39 FP:30 FN:14 TN:45
- epoch 58: auc=0.7195, f1=0.6364, recall=0.6604, precision=0.6140, score=0.6650, precisionAware=0.6418  TP:35 FP:22 FN:18 TN:53
- epoch 55: auc=0.7182, f1=0.6087, recall=0.6604, precision=0.5645, score=0.6564, precisionAware=0.6085  TP:35 FP:27 FN:18 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7155, f1=0.6214, recall=0.6038, precision=0.6400, precisionAware=0.6495, composite=0.6314  TP:32 FP:18 FN:21 TN:57
- epoch 52: auc=0.7160, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6491, composite=0.6431  TP:33 FP:19 FN:20 TN:56
- epoch 59: auc=0.7250, f1=0.6296, recall=0.6415, precision=0.6182, precisionAware=0.6430, composite=0.6546  TP:34 FP:21 FN:19 TN:54
- epoch 58: auc=0.7195, f1=0.6364, recall=0.6604, precision=0.6140, precisionAware=0.6418, composite=0.6650  TP:35 FP:22 FN:18 TN:53
