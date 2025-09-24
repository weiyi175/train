# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 3 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7326
- F1: 0.0000
- Recall: 0.0000
- Precision: 0.0000
- Composite Score: 0.1465 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.1465 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 184
- TN: 170

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6459, f1=0.1875, recall=0.1111, precision=0.6000, score=0.2410, precisionAware=0.4854  TP:6 FP:4 FN:48 TN:70
- epoch 3: auc=0.6557, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1311, precisionAware=0.1311  TP:0 FP:0 FN:54 TN:74
- epoch 2: auc=0.6454, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1291, precisionAware=0.1291  TP:0 FP:0 FN:54 TN:74

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6459, f1=0.1875, recall=0.1111, precision=0.6000, precisionAware=0.4854, composite=0.2410  TP:6 FP:4 FN:48 TN:70
- epoch 3: auc=0.6557, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1311, composite=0.1311  TP:0 FP:0 FN:54 TN:74
- epoch 2: auc=0.6454, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1291, composite=0.1291  TP:0 FP:0 FN:54 TN:74
