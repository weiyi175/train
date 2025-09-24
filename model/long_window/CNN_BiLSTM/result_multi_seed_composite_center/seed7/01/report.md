# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed7 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 7 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6140
- F1: 0.6417
- Recall: 0.7446
- Precision: 0.5638
- Composite Score: 0.6876 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5972 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 137
- FP: 106
- FN: 47
- TN: 64

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 46: auc=0.7715, f1=0.6829, recall=0.7778, precision=0.6087, score=0.7481, precisionAware=0.6635  TP:42 FP:27 FN:12 TN:47
- epoch 70: auc=0.7235, f1=0.6471, recall=0.8148, precision=0.5366, score=0.7462, precisionAware=0.6071  TP:44 FP:38 FN:10 TN:36
- epoch 57: auc=0.7285, f1=0.6719, recall=0.7963, precision=0.5811, score=0.7454, precisionAware=0.6378  TP:43 FP:31 FN:11 TN:43
- epoch 55: auc=0.7467, f1=0.6829, recall=0.7778, precision=0.6087, score=0.7431, precisionAware=0.6586  TP:42 FP:27 FN:12 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 25: auc=0.7865, f1=0.5977, recall=0.4815, precision=0.7879, precisionAware=0.7306, composite=0.5774  TP:26 FP:7 FN:28 TN:67
- epoch 34: auc=0.7770, f1=0.6800, recall=0.6296, precision=0.7391, precisionAware=0.7290, composite=0.6742  TP:34 FP:12 FN:20 TN:62
- epoch 24: auc=0.7710, f1=0.6923, recall=0.6667, precision=0.7200, precisionAware=0.7219, composite=0.6952  TP:36 FP:14 FN:18 TN:60
- epoch 28: auc=0.7813, f1=0.6316, recall=0.5556, precision=0.7317, precisionAware=0.7116, composite=0.6235  TP:30 FP:11 FN:24 TN:63
