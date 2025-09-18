# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed17 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 17 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7108
- F1: 0.5975
- Recall: 0.6857
- Precision: 0.5294
- Composite Score: 0.6643 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5861 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 72
- FP: 64
- FN: 33
- TN: 87

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7303, f1=0.6667, recall=0.7547, precision=0.5970, score=0.7234, precisionAware=0.6446  TP:40 FP:27 FN:13 TN:48
- epoch 37: auc=0.7137, f1=0.6557, recall=0.7547, precision=0.5797, score=0.7168, precisionAware=0.6293  TP:40 FP:29 FN:13 TN:46
- epoch 52: auc=0.7057, f1=0.6446, recall=0.7358, precision=0.5735, score=0.7024, precisionAware=0.6213  TP:39 FP:29 FN:14 TN:46
- epoch 42: auc=0.7394, f1=0.6542, recall=0.6604, precision=0.6481, score=0.6743, precisionAware=0.6682  TP:35 FP:19 FN:18 TN:56

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 42: auc=0.7394, f1=0.6542, recall=0.6604, precision=0.6481, precisionAware=0.6682, composite=0.6743  TP:35 FP:19 FN:18 TN:56
- epoch 34: auc=0.7258, f1=0.6122, recall=0.5660, precision=0.6667, precisionAware=0.6622, composite=0.6118  TP:30 FP:15 FN:23 TN:60
- epoch 44: auc=0.7336, f1=0.6481, recall=0.6604, precision=0.6364, precisionAware=0.6593, composite=0.6714  TP:35 FP:20 FN:18 TN:55
- epoch 29: auc=0.7346, f1=0.5806, recall=0.5094, precision=0.6750, precisionAware=0.6586, composite=0.5758  TP:27 FP:13 FN:26 TN:62
