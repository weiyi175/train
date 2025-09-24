# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed8 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 8 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6359
- F1: 0.5952
- Recall: 0.6033
- Precision: 0.5873
- Composite Score: 0.6074 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5994 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 111
- FP: 78
- FN: 73
- TN: 92

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7465, f1=0.6519, recall=0.8148, precision=0.5432, score=0.7523, precisionAware=0.6165  TP:44 FP:37 FN:10 TN:37
- epoch 20: auc=0.7525, f1=0.6721, recall=0.7593, precision=0.6029, score=0.7318, precisionAware=0.6536  TP:41 FP:27 FN:13 TN:47
- epoch 43: auc=0.7675, f1=0.6613, recall=0.7593, precision=0.5857, score=0.7315, precisionAware=0.6447  TP:41 FP:29 FN:13 TN:45
- epoch 36: auc=0.7803, f1=0.6780, recall=0.7407, precision=0.6250, score=0.7298, precisionAware=0.6719  TP:40 FP:24 FN:14 TN:50

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 30: auc=0.7853, f1=0.5679, recall=0.4259, precision=0.8519, precisionAware=0.7534, composite=0.5404  TP:23 FP:4 FN:31 TN:70
- epoch 32: auc=0.7868, f1=0.6237, recall=0.5370, precision=0.7436, precisionAware=0.7162, composite=0.6130  TP:29 FP:10 FN:25 TN:64
- epoch 55: auc=0.7823, f1=0.5581, recall=0.4444, precision=0.7500, precisionAware=0.6989, composite=0.5461  TP:24 FP:8 FN:30 TN:66
- epoch 28: auc=0.7588, f1=0.6535, recall=0.6111, precision=0.7021, precisionAware=0.6989, composite=0.6533  TP:33 FP:14 FN:21 TN:60
