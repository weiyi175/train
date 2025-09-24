# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 50 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7189
- F1: 0.6391
- Recall: 0.5870
- Precision: 0.7013
- Composite Score: 0.6290 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6861 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 108
- FP: 46
- FN: 76
- TN: 124

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 36: auc=0.7790, f1=0.6774, recall=0.7778, precision=0.6000, score=0.7479, precisionAware=0.6590  TP:42 FP:28 FN:12 TN:46
- epoch 40: auc=0.7993, f1=0.6852, recall=0.6852, precision=0.6852, score=0.7080, precisionAware=0.7080  TP:37 FP:17 FN:17 TN:57
- epoch 27: auc=0.8036, f1=0.6923, recall=0.6667, precision=0.7200, score=0.7017, precisionAware=0.7284  TP:36 FP:14 FN:18 TN:60
- epoch 34: auc=0.7900, f1=0.6923, recall=0.6667, precision=0.7200, score=0.6990, precisionAware=0.7257  TP:36 FP:14 FN:18 TN:60

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 23: auc=0.7940, f1=0.6207, recall=0.5000, precision=0.8182, precisionAware=0.7541, composite=0.5950  TP:27 FP:6 FN:27 TN:68
- epoch 30: auc=0.7953, f1=0.5783, recall=0.4444, precision=0.8276, precisionAware=0.7463, composite=0.5548  TP:24 FP:5 FN:30 TN:69
- epoch 24: auc=0.7950, f1=0.6374, recall=0.5370, precision=0.7838, precisionAware=0.7421, composite=0.6187  TP:29 FP:8 FN:25 TN:66
- epoch 18: auc=0.7628, f1=0.6292, recall=0.5185, precision=0.8000, precisionAware=0.7413, composite=0.6006  TP:28 FP:7 FN:26 TN:67
