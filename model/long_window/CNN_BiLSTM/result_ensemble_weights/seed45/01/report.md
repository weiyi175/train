# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed45 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 45 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6827
- F1: 0.6821
- Recall: 0.7228
- Precision: 0.6456
- Composite Score: 0.7026 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6640 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 133
- FP: 73
- FN: 51
- TN: 97

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5104, f1=0.5856, recall=1.0000, precision=0.4141, score=0.7778, precisionAware=0.4848  TP:53 FP:75 FN:0 TN:0
- epoch 60: auc=0.6981, f1=0.6271, recall=0.6981, precision=0.5692, score=0.6768, precisionAware=0.6124  TP:37 FP:28 FN:16 TN:47
- epoch 59: auc=0.7117, f1=0.6316, recall=0.6792, precision=0.5902, score=0.6714, precisionAware=0.6269  TP:36 FP:25 FN:17 TN:50
- epoch 58: auc=0.7029, f1=0.6016, recall=0.6981, precision=0.5286, score=0.6701, precisionAware=0.5854  TP:37 FP:33 FN:16 TN:42

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7117, f1=0.6316, recall=0.6792, precision=0.5902, precisionAware=0.6269, composite=0.6714  TP:36 FP:25 FN:17 TN:50
- epoch 53: auc=0.7024, f1=0.6207, recall=0.6792, precision=0.5714, precisionAware=0.6124, composite=0.6663  TP:36 FP:27 FN:17 TN:48
- epoch 60: auc=0.6981, f1=0.6271, recall=0.6981, precision=0.5692, precisionAware=0.6124, composite=0.6768  TP:37 FP:28 FN:16 TN:47
- epoch 52: auc=0.7016, f1=0.6050, recall=0.6792, precision=0.5455, precisionAware=0.5946, composite=0.6615  TP:36 FP:30 FN:17 TN:45
