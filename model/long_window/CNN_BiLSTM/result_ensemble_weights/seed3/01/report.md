# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed3 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 3 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6818
- F1: 0.6650
- Recall: 0.7174
- Precision: 0.6197
- Composite Score: 0.6946 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6457 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 132
- FP: 81
- FN: 52
- TN: 89

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 41: auc=0.7039, f1=0.6567, recall=0.8302, precision=0.5432, score=0.7529, precisionAware=0.6094  TP:44 FP:37 FN:9 TN:38
- epoch 40: auc=0.7044, f1=0.6565, recall=0.8113, precision=0.5513, score=0.7435, precisionAware=0.6135  TP:43 FP:35 FN:10 TN:40
- epoch 42: auc=0.7026, f1=0.6565, recall=0.8113, precision=0.5513, score=0.7431, precisionAware=0.6131  TP:43 FP:35 FN:10 TN:40
- epoch 55: auc=0.6969, f1=0.6412, recall=0.7925, precision=0.5385, score=0.7280, precisionAware=0.6010  TP:42 FP:36 FN:11 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.7026, f1=0.6560, recall=0.7736, precision=0.5694, precisionAware=0.6221, composite=0.7241  TP:41 FP:31 FN:12 TN:44
- epoch 48: auc=0.7067, f1=0.6508, recall=0.7736, precision=0.5616, precisionAware=0.6174, composite=0.7234  TP:41 FP:32 FN:12 TN:43
- epoch 52: auc=0.7031, f1=0.6508, recall=0.7736, precision=0.5616, precisionAware=0.6167, composite=0.7227  TP:41 FP:32 FN:12 TN:43
- epoch 51: auc=0.7047, f1=0.6452, recall=0.7547, precision=0.5634, precisionAware=0.6162, composite=0.7118  TP:40 FP:31 FN:13 TN:44
