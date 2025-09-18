# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed4 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 4 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7372
- F1: 0.6387
- Recall: 0.7238
- Precision: 0.5714
- Composite Score: 0.7010 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6248 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 76
- FP: 57
- FN: 29
- TN: 94

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7743, f1=0.6897, recall=0.7547, precision=0.6349, score=0.7391, precisionAware=0.6792  TP:40 FP:23 FN:13 TN:52
- epoch 49: auc=0.7608, f1=0.6612, recall=0.7547, precision=0.5882, score=0.7279, precisionAware=0.6446  TP:40 FP:28 FN:13 TN:47
- epoch 56: auc=0.7384, f1=0.6406, recall=0.7736, precision=0.5467, score=0.7267, precisionAware=0.6132  TP:41 FP:34 FN:12 TN:41
- epoch 31: auc=0.7253, f1=0.6612, recall=0.7547, precision=0.5882, score=0.7208, precisionAware=0.6375  TP:40 FP:28 FN:13 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7821, f1=0.6733, recall=0.6415, precision=0.7083, precisionAware=0.7126, composite=0.6792  TP:34 FP:14 FN:19 TN:61
- epoch 55: auc=0.7930, f1=0.6602, recall=0.6415, precision=0.6800, precisionAware=0.6966, composite=0.6774  TP:34 FP:16 FN:19 TN:59
- epoch 59: auc=0.7552, f1=0.6400, recall=0.6038, precision=0.6809, precisionAware=0.6835, composite=0.6449  TP:32 FP:15 FN:21 TN:60
- epoch 36: auc=0.7542, f1=0.6538, recall=0.6415, precision=0.6667, precisionAware=0.6803, composite=0.6678  TP:34 FP:17 FN:19 TN:58
