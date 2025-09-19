# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed35 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 35 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6910
- F1: 0.6307
- Recall: 0.6359
- Precision: 0.6257
- Composite Score: 0.6453 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6402 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 117
- FP: 70
- FN: 67
- TN: 100

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.7092, f1=0.6560, recall=0.7736, precision=0.5694, score=0.7254, precisionAware=0.6234  TP:41 FP:31 FN:12 TN:44
- epoch 52: auc=0.7235, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7141, precisionAware=0.6145  TP:40 FP:32 FN:13 TN:43
- epoch 57: auc=0.7406, f1=0.6500, recall=0.7358, precision=0.5821, score=0.7111, precisionAware=0.6342  TP:39 FP:28 FN:14 TN:47
- epoch 53: auc=0.7303, f1=0.6555, recall=0.7358, precision=0.5909, score=0.7106, precisionAware=0.6382  TP:39 FP:27 FN:14 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 6: auc=0.5919, f1=0.1071, recall=0.0566, precision=1.0000, precisionAware=0.6505, composite=0.1788  TP:3 FP:0 FN:50 TN:75
- epoch 58: auc=0.7464, f1=0.6552, recall=0.7170, precision=0.6032, precisionAware=0.6474, composite=0.7043  TP:38 FP:25 FN:15 TN:50
- epoch 60: auc=0.7431, f1=0.6552, recall=0.7170, precision=0.6032, precisionAware=0.6468, composite=0.7037  TP:38 FP:25 FN:15 TN:50
- epoch 56: auc=0.7396, f1=0.6496, recall=0.7170, precision=0.5938, precisionAware=0.6397, composite=0.7013  TP:38 FP:26 FN:15 TN:49
