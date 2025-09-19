# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed37 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 37 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7129
- F1: 0.6974
- Recall: 0.7391
- Precision: 0.6602
- Composite Score: 0.7214 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6819 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 136
- FP: 70
- FN: 48
- TN: 100

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7467, f1=0.6720, recall=0.7925, precision=0.5833, score=0.7472, precisionAware=0.6426  TP:42 FP:30 FN:11 TN:45
- epoch 58: auc=0.7434, f1=0.6614, recall=0.7925, precision=0.5676, score=0.7433, precisionAware=0.6309  TP:42 FP:32 FN:11 TN:43
- epoch 56: auc=0.7454, f1=0.6560, recall=0.7736, precision=0.5694, score=0.7327, precisionAware=0.6306  TP:41 FP:31 FN:12 TN:44
- epoch 59: auc=0.7457, f1=0.6612, recall=0.7547, precision=0.5882, score=0.7248, precisionAware=0.6416  TP:40 FP:28 FN:13 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7379, f1=0.6610, recall=0.7358, precision=0.6000, precisionAware=0.6459, composite=0.7138  TP:39 FP:26 FN:14 TN:49
- epoch 57: auc=0.7467, f1=0.6720, recall=0.7925, precision=0.5833, precisionAware=0.6426, composite=0.7472  TP:42 FP:30 FN:11 TN:45
- epoch 59: auc=0.7457, f1=0.6612, recall=0.7547, precision=0.5882, precisionAware=0.6416, composite=0.7248  TP:40 FP:28 FN:13 TN:47
- epoch 54: auc=0.7396, f1=0.6496, recall=0.7170, precision=0.5938, precisionAware=0.6397, composite=0.7013  TP:38 FP:26 FN:15 TN:49
