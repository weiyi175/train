# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed25 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 25 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6674
- F1: 0.6324
- Recall: 0.6359
- Precision: 0.6290
- Composite Score: 0.6411 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6377 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 117
- FP: 69
- FN: 67
- TN: 101

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.4848, f1=0.5856, recall=1.0000, precision=0.4141, score=0.7726, precisionAware=0.4797  TP:53 FP:75 FN:0 TN:0
- epoch 58: auc=0.7220, f1=0.6195, recall=0.6604, precision=0.5833, score=0.6604, precisionAware=0.6219  TP:35 FP:25 FN:18 TN:50
- epoch 57: auc=0.7185, f1=0.6195, recall=0.6604, precision=0.5833, score=0.6597, precisionAware=0.6212  TP:35 FP:25 FN:18 TN:50
- epoch 53: auc=0.7009, f1=0.6034, recall=0.6604, precision=0.5556, score=0.6514, precisionAware=0.5990  TP:35 FP:28 FN:18 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7308, f1=0.6111, recall=0.6226, precision=0.6000, precisionAware=0.6295, composite=0.6408  TP:33 FP:22 FN:20 TN:53
- epoch 59: auc=0.7270, f1=0.6126, recall=0.6415, precision=0.5862, precisionAware=0.6223, composite=0.6499  TP:34 FP:24 FN:19 TN:51
- epoch 58: auc=0.7220, f1=0.6195, recall=0.6604, precision=0.5833, precisionAware=0.6219, composite=0.6604  TP:35 FP:25 FN:18 TN:50
- epoch 57: auc=0.7185, f1=0.6195, recall=0.6604, precision=0.5833, precisionAware=0.6212, composite=0.6597  TP:35 FP:25 FN:18 TN:50
