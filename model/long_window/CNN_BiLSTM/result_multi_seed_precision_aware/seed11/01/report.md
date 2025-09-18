# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed11 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 11 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7388
- F1: 0.6180
- Recall: 0.6857
- Precision: 0.5625
- Composite Score: 0.6760 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6144 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 72
- FP: 56
- FN: 33
- TN: 95

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7919, f1=0.6842, recall=0.7358, precision=0.6393, score=0.7316, precisionAware=0.6833  TP:39 FP:22 FN:14 TN:53
- epoch 60: auc=0.7882, f1=0.6486, recall=0.6792, precision=0.6207, score=0.6919, precisionAware=0.6626  TP:36 FP:22 FN:17 TN:53
- epoch 52: auc=0.7826, f1=0.6372, recall=0.6792, precision=0.6000, score=0.6873, precisionAware=0.6477  TP:36 FP:24 FN:17 TN:51
- epoch 46: auc=0.7723, f1=0.6422, recall=0.6604, precision=0.6250, score=0.6773, precisionAware=0.6596  TP:35 FP:21 FN:18 TN:54

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 54: auc=0.8191, f1=0.6408, recall=0.6226, precision=0.6600, precisionAware=0.6861, composite=0.6674  TP:33 FP:17 FN:20 TN:58
- epoch 55: auc=0.7919, f1=0.6842, recall=0.7358, precision=0.6393, precisionAware=0.6833, composite=0.7316  TP:39 FP:22 FN:14 TN:53
- epoch 47: auc=0.8008, f1=0.5957, recall=0.5283, precision=0.6829, precisionAware=0.6803, composite=0.6030  TP:28 FP:13 FN:25 TN:62
- epoch 45: auc=0.7945, f1=0.6122, recall=0.5660, precision=0.6667, precisionAware=0.6759, composite=0.6256  TP:30 FP:15 FN:23 TN:60
