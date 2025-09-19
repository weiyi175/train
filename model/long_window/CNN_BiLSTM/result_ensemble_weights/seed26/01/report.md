# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed26 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 26 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7116
- F1: 0.6559
- Recall: 0.6630
- Precision: 0.6489
- Composite Score: 0.6706 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6636 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 66
- FN: 62
- TN: 104

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7663, f1=0.6903, recall=0.7358, precision=0.6500, score=0.7283, precisionAware=0.6853  TP:39 FP:21 FN:14 TN:54
- epoch 59: auc=0.7590, f1=0.6903, recall=0.7358, precision=0.6500, score=0.7268, precisionAware=0.6839  TP:39 FP:21 FN:14 TN:54
- epoch 47: auc=0.7487, f1=0.6726, recall=0.7170, precision=0.6333, score=0.7100, precisionAware=0.6682  TP:38 FP:22 FN:15 TN:53
- epoch 52: auc=0.7572, f1=0.6496, recall=0.7170, precision=0.5938, score=0.7048, precisionAware=0.6432  TP:38 FP:26 FN:15 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7663, f1=0.6903, recall=0.7358, precision=0.6500, precisionAware=0.6853, composite=0.7283  TP:39 FP:21 FN:14 TN:54
- epoch 59: auc=0.7590, f1=0.6903, recall=0.7358, precision=0.6500, precisionAware=0.6839, composite=0.7268  TP:39 FP:21 FN:14 TN:54
- epoch 56: auc=0.7613, f1=0.6667, recall=0.6792, precision=0.6545, precisionAware=0.6795, composite=0.6919  TP:36 FP:19 FN:17 TN:56
- epoch 47: auc=0.7487, f1=0.6726, recall=0.7170, precision=0.6333, precisionAware=0.6682, composite=0.7100  TP:38 FP:22 FN:15 TN:53
