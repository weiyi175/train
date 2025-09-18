# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed16 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 16 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7281
- F1: 0.5333
- Recall: 0.4571
- Precision: 0.6400
- Composite Score: 0.5342 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6256 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 48
- FP: 27
- FN: 57
- TN: 124

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.7504, f1=0.5825, recall=0.5660, precision=0.6000, score=0.6079, precisionAware=0.6248  TP:30 FP:20 FN:23 TN:55
- epoch 36: auc=0.7472, f1=0.5859, recall=0.5472, precision=0.6304, score=0.5988, precisionAware=0.6404  TP:29 FP:17 FN:24 TN:58
- epoch 38: auc=0.7381, f1=0.5800, recall=0.5472, precision=0.6170, score=0.5952, precisionAware=0.6301  TP:29 FP:18 FN:24 TN:57
- epoch 42: auc=0.7597, f1=0.5773, recall=0.5283, precision=0.6364, score=0.5893, precisionAware=0.6433  TP:28 FP:16 FN:25 TN:59

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 37: auc=0.7592, f1=0.5495, recall=0.4717, precision=0.6579, precisionAware=0.6456, composite=0.5525  TP:25 FP:13 FN:28 TN:62
- epoch 42: auc=0.7597, f1=0.5773, recall=0.5283, precision=0.6364, precisionAware=0.6433, composite=0.5893  TP:28 FP:16 FN:25 TN:59
- epoch 30: auc=0.7409, f1=0.5495, recall=0.4717, precision=0.6579, precisionAware=0.6420, composite=0.5489  TP:25 FP:13 FN:28 TN:62
- epoch 36: auc=0.7472, f1=0.5859, recall=0.5472, precision=0.6304, precisionAware=0.6404, composite=0.5988  TP:29 FP:17 FN:24 TN:58
