# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed9 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 9 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6723
- F1: 0.6910
- Recall: 0.7717
- Precision: 0.6256
- Composite Score: 0.7276 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6545 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 142
- FP: 85
- FN: 42
- TN: 85

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7993, f1=0.7673, recall=0.9683, precision=0.6354, score=0.8742, precisionAware=0.7078  TP:61 FP:35 FN:2 TN:30
- epoch 48: auc=0.8234, f1=0.7755, recall=0.9048, precision=0.6786, score=0.8497, precisionAware=0.7366  TP:57 FP:27 FN:6 TN:38
- epoch 66: auc=0.8051, f1=0.7484, recall=0.9206, precision=0.6304, score=0.8459, precisionAware=0.7008  TP:58 FP:34 FN:5 TN:31
- epoch 45: auc=0.8078, f1=0.7832, recall=0.8889, precision=0.7000, score=0.8410, precisionAware=0.7465  TP:56 FP:24 FN:7 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 45: auc=0.8078, f1=0.7832, recall=0.8889, precision=0.7000, precisionAware=0.7465, composite=0.8410  TP:56 FP:24 FN:7 TN:41
- epoch 34: auc=0.8020, f1=0.6949, recall=0.6508, precision=0.7455, precisionAware=0.7416, composite=0.6943  TP:41 FP:14 FN:22 TN:51
- epoch 32: auc=0.7995, f1=0.7200, recall=0.7143, precision=0.7258, precisionAware=0.7388, composite=0.7330  TP:45 FP:17 FN:18 TN:48
- epoch 63: auc=0.7980, f1=0.7591, recall=0.8254, precision=0.7027, precisionAware=0.7387, composite=0.8000  TP:52 FP:22 FN:11 TN:43
