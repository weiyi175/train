# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed7 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 7 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6893
- F1: 0.6702
- Recall: 0.6902
- Precision: 0.6513
- Composite Score: 0.6840 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6646 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 127
- FP: 68
- FN: 57
- TN: 102

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7610, f1=0.6508, recall=0.7736, precision=0.5616, score=0.7342, precisionAware=0.6283  TP:41 FP:32 FN:12 TN:43
- epoch 52: auc=0.7577, f1=0.6610, recall=0.7358, precision=0.6000, score=0.7178, precisionAware=0.6499  TP:39 FP:26 FN:14 TN:49
- epoch 62: auc=0.7562, f1=0.6609, recall=0.7170, precision=0.6129, score=0.7080, precisionAware=0.6560  TP:38 FP:24 FN:15 TN:51
- epoch 38: auc=0.7376, f1=0.6341, recall=0.7358, precision=0.5571, score=0.7057, precisionAware=0.6163  TP:39 FP:31 FN:14 TN:44

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 62: auc=0.7562, f1=0.6609, recall=0.7170, precision=0.6129, precisionAware=0.6560, composite=0.7080  TP:38 FP:24 FN:15 TN:51
- epoch 61: auc=0.7758, f1=0.6364, recall=0.6604, precision=0.6140, precisionAware=0.6531, composite=0.6763  TP:35 FP:22 FN:18 TN:53
- epoch 52: auc=0.7577, f1=0.6610, recall=0.7358, precision=0.6000, precisionAware=0.6499, composite=0.7178  TP:39 FP:26 FN:14 TN:49
- epoch 66: auc=0.7595, f1=0.6372, recall=0.6792, precision=0.6000, precisionAware=0.6430, composite=0.6827  TP:36 FP:24 FN:17 TN:51
