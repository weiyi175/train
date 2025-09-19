# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed4 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 4 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6546
- F1: 0.6396
- Recall: 0.6848
- Precision: 0.6000
- Composite Score: 0.6652 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6228 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 126
- FP: 84
- FN: 58
- TN: 86

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7079, f1=0.6290, recall=0.7358, precision=0.5493, score=0.6982, precisionAware=0.6049  TP:39 FP:32 FN:14 TN:43
- epoch 54: auc=0.7074, f1=0.6290, recall=0.7358, precision=0.5493, score=0.6981, precisionAware=0.6048  TP:39 FP:32 FN:14 TN:43
- epoch 55: auc=0.7104, f1=0.6240, recall=0.7358, precision=0.5417, score=0.6972, precisionAware=0.6001  TP:39 FP:33 FN:14 TN:42
- epoch 52: auc=0.7031, f1=0.6190, recall=0.7358, precision=0.5342, score=0.6943, precisionAware=0.5935  TP:39 FP:34 FN:14 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 7: auc=0.6065, f1=0.1404, recall=0.0755, precision=1.0000, precisionAware=0.6634, composite=0.2011  TP:4 FP:0 FN:49 TN:75
- epoch 59: auc=0.7125, f1=0.6207, recall=0.6792, precision=0.5714, precisionAware=0.6144, composite=0.6683  TP:36 FP:27 FN:17 TN:48
- epoch 58: auc=0.7057, f1=0.6281, recall=0.7170, precision=0.5588, precisionAware=0.6090, composite=0.6881  TP:38 FP:30 FN:15 TN:45
- epoch 56: auc=0.7079, f1=0.6290, recall=0.7358, precision=0.5493, precisionAware=0.6049, composite=0.6982  TP:39 FP:32 FN:14 TN:43
