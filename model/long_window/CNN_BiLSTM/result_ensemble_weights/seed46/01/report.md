# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed46 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 46 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7284
- F1: 0.6207
- Recall: 0.5380
- Precision: 0.7333
- Composite Score: 0.6009 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6985 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 99
- FP: 36
- FN: 85
- TN: 134

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.7411, f1=0.6379, recall=0.6981, precision=0.5873, score=0.6887, precisionAware=0.6333  TP:37 FP:26 FN:16 TN:49
- epoch 58: auc=0.7381, f1=0.6325, recall=0.6981, precision=0.5781, score=0.6864, precisionAware=0.6264  TP:37 FP:27 FN:16 TN:48
- epoch 52: auc=0.7394, f1=0.6486, recall=0.6792, precision=0.6207, score=0.6821, precisionAware=0.6528  TP:36 FP:22 FN:17 TN:53
- epoch 48: auc=0.7399, f1=0.6429, recall=0.6792, precision=0.6102, score=0.6805, precisionAware=0.6459  TP:36 FP:23 FN:17 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.7394, f1=0.6486, recall=0.6792, precision=0.6207, precisionAware=0.6528, composite=0.6821  TP:36 FP:22 FN:17 TN:53
- epoch 48: auc=0.7399, f1=0.6429, recall=0.6792, precision=0.6102, precisionAware=0.6459, composite=0.6805  TP:36 FP:23 FN:17 TN:52
- epoch 56: auc=0.7572, f1=0.6095, recall=0.6038, precision=0.6154, precisionAware=0.6420, composite=0.6362  TP:32 FP:20 FN:21 TN:55
- epoch 55: auc=0.7542, f1=0.5941, recall=0.5660, precision=0.6250, precisionAware=0.6416, composite=0.6121  TP:30 FP:18 FN:23 TN:57
