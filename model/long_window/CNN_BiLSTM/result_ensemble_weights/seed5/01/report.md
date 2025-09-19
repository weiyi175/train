# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed5 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 5 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7096
- F1: 0.6721
- Recall: 0.6739
- Precision: 0.6703
- Composite Score: 0.6805 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6787 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 124
- FP: 61
- FN: 60
- TN: 109

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5102, f1=0.5763, recall=0.9623, precision=0.4113, score=0.7561, precisionAware=0.4806  TP:51 FP:73 FN:2 TN:2
- epoch 59: auc=0.7416, f1=0.6435, recall=0.6981, precision=0.5968, score=0.6904, precisionAware=0.6398  TP:37 FP:25 FN:16 TN:50
- epoch 60: auc=0.7409, f1=0.6435, recall=0.6981, precision=0.5968, score=0.6903, precisionAware=0.6396  TP:37 FP:25 FN:16 TN:50
- epoch 56: auc=0.7547, f1=0.6606, recall=0.6792, precision=0.6429, score=0.6887, precisionAware=0.6705  TP:36 FP:20 FN:17 TN:55

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7547, f1=0.6606, recall=0.6792, precision=0.6429, precisionAware=0.6705, composite=0.6887  TP:36 FP:20 FN:17 TN:55
- epoch 57: auc=0.7547, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6568, composite=0.6508  TP:33 FP:19 FN:20 TN:56
- epoch 50: auc=0.7522, f1=0.6364, recall=0.6604, precision=0.6140, precisionAware=0.6484, composite=0.6715  TP:35 FP:22 FN:18 TN:53
- epoch 55: auc=0.7535, f1=0.6372, recall=0.6792, precision=0.6000, precisionAware=0.6418, composite=0.6815  TP:36 FP:24 FN:17 TN:51
