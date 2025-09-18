# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed6 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 6 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7032
- F1: 0.5729
- Recall: 0.5429
- Precision: 0.6064
- Composite Score: 0.5839 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6157 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 37
- FN: 48
- TN: 114

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 48: auc=0.7296, f1=0.6457, recall=0.7736, precision=0.5541, score=0.7264, precisionAware=0.6166  TP:41 FP:33 FN:12 TN:42
- epoch 51: auc=0.7396, f1=0.6667, recall=0.7547, precision=0.5970, score=0.7253, precisionAware=0.6464  TP:40 FP:27 FN:13 TN:48
- epoch 46: auc=0.7326, f1=0.6555, recall=0.7358, precision=0.5909, score=0.7111, precisionAware=0.6386  TP:39 FP:27 FN:14 TN:48
- epoch 37: auc=0.7447, f1=0.6446, recall=0.7358, precision=0.5735, score=0.7102, precisionAware=0.6291  TP:39 FP:29 FN:14 TN:46

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7587, f1=0.6000, recall=0.5660, precision=0.6383, precisionAware=0.6509, composite=0.6148  TP:30 FP:17 FN:23 TN:58
- epoch 27: auc=0.7167, f1=0.6355, recall=0.6415, precision=0.6296, precisionAware=0.6488, composite=0.6548  TP:34 FP:20 FN:19 TN:55
- epoch 41: auc=0.7197, f1=0.6549, recall=0.6981, precision=0.6167, precisionAware=0.6487, composite=0.6895  TP:37 FP:23 FN:16 TN:52
- epoch 51: auc=0.7396, f1=0.6667, recall=0.7547, precision=0.5970, precisionAware=0.6464, composite=0.7253  TP:40 FP:27 FN:13 TN:48
