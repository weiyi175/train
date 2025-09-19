# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed36 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 36 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6728
- F1: 0.6017
- Recall: 0.5707
- Precision: 0.6364
- Composite Score: 0.6004 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6333 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 105
- FP: 60
- FN: 79
- TN: 110

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7572, f1=0.6667, recall=0.7170, precision=0.6230, score=0.7099, precisionAware=0.6629  TP:38 FP:23 FN:15 TN:52
- epoch 55: auc=0.7648, f1=0.6789, recall=0.6981, precision=0.6607, score=0.7057, precisionAware=0.6870  TP:37 FP:19 FN:16 TN:56
- epoch 47: auc=0.7595, f1=0.6727, recall=0.6981, precision=0.6491, score=0.7028, precisionAware=0.6783  TP:37 FP:20 FN:16 TN:55
- epoch 48: auc=0.7628, f1=0.6667, recall=0.6981, precision=0.6379, score=0.7016, precisionAware=0.6715  TP:37 FP:21 FN:16 TN:54

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7620, f1=0.6733, recall=0.6415, precision=0.7083, precisionAware=0.7085, composite=0.6751  TP:34 FP:14 FN:19 TN:61
- epoch 57: auc=0.7603, f1=0.6602, recall=0.6415, precision=0.6800, precisionAware=0.6901, composite=0.6709  TP:34 FP:16 FN:19 TN:59
- epoch 52: auc=0.7582, f1=0.6602, recall=0.6415, precision=0.6800, precisionAware=0.6897, composite=0.6705  TP:34 FP:16 FN:19 TN:59
- epoch 56: auc=0.7633, f1=0.6729, recall=0.6792, precision=0.6667, precisionAware=0.6879, composite=0.6941  TP:36 FP:18 FN:17 TN:57
