# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed20 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 20 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7191
- F1: 0.6175
- Recall: 0.6381
- Precision: 0.5982
- Composite Score: 0.6481 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6282 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 67
- FP: 45
- FN: 38
- TN: 106

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 51: auc=0.6810, f1=0.6061, recall=0.7547, precision=0.5063, score=0.6954, precisionAware=0.5712  TP:40 FP:39 FN:13 TN:36
- epoch 31: auc=0.6848, f1=0.6000, recall=0.6792, precision=0.5373, score=0.6566, precisionAware=0.5856  TP:36 FP:31 FN:17 TN:44
- epoch 48: auc=0.7165, f1=0.5913, recall=0.6415, precision=0.5484, score=0.6414, precisionAware=0.5949  TP:34 FP:28 FN:19 TN:47
- epoch 41: auc=0.7026, f1=0.5913, recall=0.6415, precision=0.5484, score=0.6387, precisionAware=0.5921  TP:34 FP:28 FN:19 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 54: auc=0.7270, f1=0.5532, recall=0.4906, precision=0.6341, precisionAware=0.6284, composite=0.5566  TP:26 FP:15 FN:27 TN:60
- epoch 59: auc=0.7258, f1=0.5825, recall=0.5660, precision=0.6000, precisionAware=0.6199, composite=0.6029  TP:30 FP:20 FN:23 TN:55
- epoch 57: auc=0.7323, f1=0.5926, recall=0.6038, precision=0.5818, precisionAware=0.6152, composite=0.6261  TP:32 FP:23 FN:21 TN:52
- epoch 44: auc=0.7185, f1=0.5849, recall=0.5849, precision=0.5849, precisionAware=0.6116, composite=0.6116  TP:31 FP:22 FN:22 TN:53
