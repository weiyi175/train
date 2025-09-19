# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed19 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 19 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6623
- F1: 0.6553
- Recall: 0.7337
- Precision: 0.5921
- Composite Score: 0.6959 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6251 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 135
- FP: 93
- FN: 49
- TN: 77

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7479, f1=0.6716, recall=0.8491, precision=0.5556, score=0.7756, precisionAware=0.6289  TP:45 FP:36 FN:8 TN:39
- epoch 56: auc=0.7396, f1=0.6719, recall=0.8113, precision=0.5733, score=0.7551, precisionAware=0.6362  TP:43 FP:32 FN:10 TN:43
- epoch 54: auc=0.7245, f1=0.6667, recall=0.8113, precision=0.5658, score=0.7506, precisionAware=0.6278  TP:43 FP:33 FN:10 TN:42
- epoch 53: auc=0.7192, f1=0.6667, recall=0.8113, precision=0.5658, score=0.7495, precisionAware=0.6267  TP:43 FP:33 FN:10 TN:42

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7517, f1=0.6833, recall=0.7736, precision=0.6119, precisionAware=0.6613, composite=0.7421  TP:41 FP:26 FN:12 TN:49
- epoch 59: auc=0.7532, f1=0.6777, recall=0.7736, precision=0.6029, precisionAware=0.6554, composite=0.7407  TP:41 FP:27 FN:12 TN:48
- epoch 57: auc=0.7464, f1=0.6721, recall=0.7736, precision=0.5942, precisionAware=0.6480, composite=0.7377  TP:41 FP:28 FN:12 TN:47
- epoch 55: auc=0.7326, f1=0.6720, recall=0.7925, precision=0.5833, precisionAware=0.6398, composite=0.7443  TP:42 FP:30 FN:11 TN:45
