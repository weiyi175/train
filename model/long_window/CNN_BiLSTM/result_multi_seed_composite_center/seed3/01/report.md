# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed3 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 3 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6825
- F1: 0.5825
- Recall: 0.4891
- Precision: 0.7200
- Composite Score: 0.5558 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6713 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 90
- FP: 35
- FN: 94
- TN: 135

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 45: auc=0.7280, f1=0.6825, recall=0.7963, precision=0.5972, score=0.7485, precisionAware=0.6490  TP:43 FP:29 FN:11 TN:45
- epoch 66: auc=0.7663, f1=0.7009, recall=0.7593, precision=0.6508, score=0.7431, precisionAware=0.6889  TP:41 FP:22 FN:13 TN:52
- epoch 64: auc=0.7683, f1=0.6783, recall=0.7222, precision=0.6393, score=0.7182, precisionAware=0.6768  TP:39 FP:22 FN:15 TN:52
- epoch 62: auc=0.7550, f1=0.6724, recall=0.7222, precision=0.6290, score=0.7138, precisionAware=0.6672  TP:39 FP:23 FN:15 TN:51

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 23: auc=0.7708, f1=0.6792, recall=0.6667, precision=0.6923, precisionAware=0.7041, composite=0.6913  TP:36 FP:16 FN:18 TN:58
- epoch 22: auc=0.7653, f1=0.6535, recall=0.6111, precision=0.7021, precisionAware=0.7002, composite=0.6546  TP:33 FP:14 FN:21 TN:60
- epoch 21: auc=0.7500, f1=0.6852, recall=0.6852, precision=0.6852, precisionAware=0.6981, composite=0.6981  TP:37 FP:17 FN:17 TN:57
- epoch 27: auc=0.7510, f1=0.6535, recall=0.6111, precision=0.7021, precisionAware=0.6973, composite=0.6518  TP:33 FP:14 FN:21 TN:60
