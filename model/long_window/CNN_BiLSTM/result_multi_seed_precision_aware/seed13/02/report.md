# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed13 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 13 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7003
- F1: 0.6404
- Recall: 0.6630
- Precision: 0.6193
- Composite Score: 0.6637 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6418 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 75
- FN: 62
- TN: 95

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 46: auc=0.6785, f1=0.5841, recall=0.6226, precision=0.5500, score=0.6222, precisionAware=0.5859  TP:33 FP:27 FN:20 TN:48
- epoch 39: auc=0.6684, f1=0.5841, recall=0.6226, precision=0.5500, score=0.6202, precisionAware=0.5839  TP:33 FP:27 FN:20 TN:48
- epoch 57: auc=0.6757, f1=0.5714, recall=0.6038, precision=0.5424, score=0.6085, precisionAware=0.5778  TP:32 FP:27 FN:21 TN:48
- epoch 36: auc=0.6702, f1=0.5517, recall=0.6038, precision=0.5079, score=0.6014, precisionAware=0.5535  TP:32 FP:31 FN:21 TN:44

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 23: auc=0.6730, f1=0.5474, recall=0.4906, precision=0.6190, precisionAware=0.6083, composite=0.5441  TP:26 FP:16 FN:27 TN:59
- epoch 22: auc=0.6813, f1=0.5417, recall=0.4906, precision=0.6047, precisionAware=0.6011, composite=0.5440  TP:26 FP:17 FN:27 TN:58
- epoch 43: auc=0.7029, f1=0.5263, recall=0.4717, precision=0.5952, precisionAware=0.5961, composite=0.5343  TP:25 FP:17 FN:28 TN:58
- epoch 27: auc=0.6830, f1=0.5263, recall=0.4717, precision=0.5952, precisionAware=0.5921, composite=0.5303  TP:25 FP:17 FN:28 TN:58
