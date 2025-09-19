# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed6 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 6 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6954
- F1: 0.6059
- Recall: 0.5598
- Precision: 0.6603
- Composite Score: 0.6007 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6510 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 103
- FP: 53
- FN: 81
- TN: 117

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7182, f1=0.6080, recall=0.7170, precision=0.5278, score=0.6845, precisionAware=0.5899  TP:38 FP:34 FN:15 TN:41
- epoch 52: auc=0.7147, f1=0.6167, recall=0.6981, precision=0.5522, score=0.6770, precisionAware=0.6041  TP:37 FP:30 FN:16 TN:45
- epoch 59: auc=0.7228, f1=0.6207, recall=0.6792, precision=0.5714, score=0.6704, precisionAware=0.6165  TP:36 FP:27 FN:17 TN:48
- epoch 51: auc=0.7185, f1=0.6102, recall=0.6792, precision=0.5538, score=0.6664, precisionAware=0.6037  TP:36 FP:29 FN:17 TN:46

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 7: auc=0.5952, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6409, composite=0.1597  TP:2 FP:0 FN:51 TN:75
- epoch 6: auc=0.5965, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6304, composite=0.1398  TP:1 FP:0 FN:52 TN:75
- epoch 59: auc=0.7228, f1=0.6207, recall=0.6792, precision=0.5714, precisionAware=0.6165, composite=0.6704  TP:36 FP:27 FN:17 TN:48
- epoch 57: auc=0.7255, f1=0.6018, recall=0.6415, precision=0.5667, precisionAware=0.6090, composite=0.6464  TP:34 FP:26 FN:19 TN:49
