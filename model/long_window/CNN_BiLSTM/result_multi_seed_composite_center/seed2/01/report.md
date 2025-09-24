# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed2 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 2 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7435
- F1: 0.6452
- Recall: 0.5978
- Precision: 0.7006
- Composite Score: 0.6412 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6926 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 110
- FP: 47
- FN: 74
- TN: 123

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7673, f1=0.7049, recall=0.7963, precision=0.6324, score=0.7631, precisionAware=0.6811  TP:43 FP:25 FN:11 TN:49
- epoch 52: auc=0.7505, f1=0.6614, recall=0.7778, precision=0.5753, score=0.7374, precisionAware=0.6362  TP:42 FP:31 FN:12 TN:43
- epoch 66: auc=0.7793, f1=0.6838, recall=0.7407, precision=0.6349, score=0.7314, precisionAware=0.6784  TP:40 FP:23 FN:14 TN:51
- epoch 54: auc=0.7840, f1=0.6610, recall=0.7222, precision=0.6094, score=0.7162, precisionAware=0.6598  TP:39 FP:25 FN:15 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7758, f1=0.5500, recall=0.4074, precision=0.8462, precisionAware=0.7432, composite=0.5239  TP:22 FP:4 FN:32 TN:70
- epoch 27: auc=0.7738, f1=0.5250, recall=0.3889, precision=0.8077, precisionAware=0.7161, composite=0.5067  TP:21 FP:5 FN:33 TN:69
- epoch 28: auc=0.7858, f1=0.6000, recall=0.5000, precision=0.7500, precisionAware=0.7122, composite=0.5872  TP:27 FP:9 FN:27 TN:65
- epoch 35: auc=0.7808, f1=0.5843, recall=0.4815, precision=0.7429, precisionAware=0.7029, composite=0.5722  TP:26 FP:9 FN:28 TN:65
