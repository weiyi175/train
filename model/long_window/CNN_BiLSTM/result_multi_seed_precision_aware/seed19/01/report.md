# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed19 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 19 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7367
- F1: 0.6250
- Recall: 0.7143
- Precision: 0.5556
- Composite Score: 0.6920 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6126 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 75
- FP: 60
- FN: 30
- TN: 91

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7713, f1=0.7080, recall=0.7547, precision=0.6667, score=0.7440, precisionAware=0.7000  TP:40 FP:20 FN:13 TN:55
- epoch 39: auc=0.7185, f1=0.6667, recall=0.7547, precision=0.5970, score=0.7211, precisionAware=0.6422  TP:40 FP:27 FN:13 TN:48
- epoch 35: auc=0.7011, f1=0.6308, recall=0.7736, precision=0.5325, score=0.7162, precisionAware=0.5957  TP:41 FP:36 FN:12 TN:39
- epoch 49: auc=0.7469, f1=0.6786, recall=0.7170, precision=0.6441, score=0.7114, precisionAware=0.6750  TP:38 FP:21 FN:15 TN:54

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7713, f1=0.7080, recall=0.7547, precision=0.6667, precisionAware=0.7000, composite=0.7440  TP:40 FP:20 FN:13 TN:55
- epoch 49: auc=0.7469, f1=0.6786, recall=0.7170, precision=0.6441, precisionAware=0.6750, composite=0.7114  TP:38 FP:21 FN:15 TN:54
- epoch 40: auc=0.7250, f1=0.6786, recall=0.7170, precision=0.6441, precisionAware=0.6706, composite=0.7071  TP:38 FP:21 FN:15 TN:54
- epoch 56: auc=0.7567, f1=0.6607, recall=0.6981, precision=0.6271, precisionAware=0.6631, composite=0.6986  TP:37 FP:22 FN:16 TN:53
