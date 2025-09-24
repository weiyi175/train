# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed4 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 4 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7225
- F1: 0.6595
- Recall: 0.6630
- Precision: 0.6559
- Composite Score: 0.6739 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6703 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 64
- FN: 62
- TN: 106

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 61: auc=0.7788, f1=0.6723, recall=0.7407, precision=0.6154, score=0.7278, precisionAware=0.6651  TP:40 FP:25 FN:14 TN:49
- epoch 50: auc=0.8068, f1=0.7103, recall=0.7037, precision=0.7170, score=0.7263, precisionAware=0.7329  TP:38 FP:15 FN:16 TN:59
- epoch 54: auc=0.7928, f1=0.6724, recall=0.7222, precision=0.6290, score=0.7214, precisionAware=0.6748  TP:39 FP:23 FN:15 TN:51
- epoch 70: auc=0.7950, f1=0.6726, recall=0.7037, precision=0.6441, score=0.7126, precisionAware=0.6828  TP:38 FP:21 FN:16 TN:53

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.8068, f1=0.7103, recall=0.7037, precision=0.7170, precisionAware=0.7329, composite=0.7263  TP:38 FP:15 FN:16 TN:59
- epoch 35: auc=0.7603, f1=0.6452, recall=0.5556, precision=0.7692, precisionAware=0.7302, composite=0.6234  TP:30 FP:9 FN:24 TN:65
- epoch 21: auc=0.7930, f1=0.6598, recall=0.5926, precision=0.7442, precisionAware=0.7286, composite=0.6528  TP:32 FP:11 FN:22 TN:63
- epoch 69: auc=0.8011, f1=0.6667, recall=0.6111, precision=0.7333, precisionAware=0.7269, composite=0.6658  TP:33 FP:12 FN:21 TN:62
