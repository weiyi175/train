# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed4 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 4 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7202
- F1: 0.6987
- Recall: 0.7500
- Precision: 0.6540
- Composite Score: 0.7287 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6807 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 138
- FP: 73
- FN: 46
- TN: 97

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 33: auc=0.7550, f1=0.6607, recall=0.6981, precision=0.6271, score=0.6983, precisionAware=0.6628  TP:37 FP:22 FN:16 TN:53
- epoch 24: auc=0.6881, f1=0.6230, recall=0.7170, precision=0.5507, score=0.6830, precisionAware=0.5999  TP:38 FP:31 FN:15 TN:44
- epoch 49: auc=0.7620, f1=0.6261, recall=0.6792, precision=0.5806, score=0.6799, precisionAware=0.6306  TP:36 FP:26 FN:17 TN:49
- epoch 60: auc=0.7424, f1=0.6154, recall=0.6792, precision=0.5625, score=0.6727, precisionAware=0.6143  TP:36 FP:28 FN:17 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 33: auc=0.7550, f1=0.6607, recall=0.6981, precision=0.6271, precisionAware=0.6628, composite=0.6983  TP:37 FP:22 FN:16 TN:53
- epoch 57: auc=0.7623, f1=0.6355, recall=0.6415, precision=0.6296, precisionAware=0.6579, composite=0.6639  TP:34 FP:20 FN:19 TN:55
- epoch 32: auc=0.7532, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6565, composite=0.6505  TP:33 FP:19 FN:20 TN:56
- epoch 39: auc=0.7595, f1=0.6000, recall=0.5660, precision=0.6383, precisionAware=0.6510, composite=0.6149  TP:30 FP:17 FN:23 TN:58
