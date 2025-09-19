# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed24 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 24 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7096
- F1: 0.7088
- Recall: 0.8533
- Precision: 0.6062
- Composite Score: 0.7812 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6577 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 157
- FP: 102
- FN: 27
- TN: 68

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5643, f1=0.5856, recall=1.0000, precision=0.4141, score=0.7885, precisionAware=0.4956  TP:53 FP:75 FN:0 TN:0
- epoch 52: auc=0.6994, f1=0.6370, recall=0.8113, precision=0.5244, score=0.7366, precisionAware=0.5932  TP:43 FP:39 FN:10 TN:36
- epoch 54: auc=0.6981, f1=0.6277, recall=0.8113, precision=0.5119, score=0.7336, precisionAware=0.5839  TP:43 FP:41 FN:10 TN:34
- epoch 55: auc=0.6916, f1=0.6277, recall=0.8113, precision=0.5119, score=0.7323, precisionAware=0.5826  TP:43 FP:41 FN:10 TN:34

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 51: auc=0.7021, f1=0.6281, recall=0.7170, precision=0.5588, precisionAware=0.6083, composite=0.6873  TP:38 FP:30 FN:15 TN:45
- epoch 52: auc=0.6994, f1=0.6370, recall=0.8113, precision=0.5244, precisionAware=0.5932, composite=0.7366  TP:43 FP:39 FN:10 TN:36
- epoch 58: auc=0.6875, f1=0.6250, recall=0.7547, precision=0.5333, precisionAware=0.5917, composite=0.7024  TP:40 FP:35 FN:13 TN:40
- epoch 50: auc=0.6896, f1=0.6190, recall=0.7358, precision=0.5342, precisionAware=0.5907, composite=0.6916  TP:39 FP:34 FN:14 TN:41
