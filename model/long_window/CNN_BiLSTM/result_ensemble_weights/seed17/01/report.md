# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed17 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 17 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7139
- F1: 0.7105
- Recall: 0.7935
- Precision: 0.6432
- Composite Score: 0.7526 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6775 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 146
- FP: 81
- FN: 38
- TN: 89

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7323, f1=0.6716, recall=0.8491, precision=0.5556, score=0.7725, precisionAware=0.6257  TP:45 FP:36 FN:8 TN:39
- epoch 55: auc=0.7351, f1=0.6875, recall=0.8302, precision=0.5867, score=0.7684, precisionAware=0.6466  TP:44 FP:31 FN:9 TN:44
- epoch 56: auc=0.7270, f1=0.6822, recall=0.8302, precision=0.5789, score=0.7652, precisionAware=0.6395  TP:44 FP:32 FN:9 TN:43
- epoch 57: auc=0.7235, f1=0.6769, recall=0.8302, precision=0.5714, score=0.7629, precisionAware=0.6335  TP:44 FP:33 FN:9 TN:42

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7351, f1=0.6875, recall=0.8302, precision=0.5867, precisionAware=0.6466, composite=0.7684  TP:44 FP:31 FN:9 TN:44
- epoch 59: auc=0.7293, f1=0.6772, recall=0.8113, precision=0.5811, precisionAware=0.6396, composite=0.7547  TP:43 FP:31 FN:10 TN:44
- epoch 56: auc=0.7270, f1=0.6822, recall=0.8302, precision=0.5789, precisionAware=0.6395, composite=0.7652  TP:44 FP:32 FN:9 TN:43
- epoch 57: auc=0.7235, f1=0.6769, recall=0.8302, precision=0.5714, precisionAware=0.6335, composite=0.7629  TP:44 FP:33 FN:9 TN:42
