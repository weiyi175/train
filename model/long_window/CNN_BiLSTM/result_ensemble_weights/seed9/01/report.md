# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed9 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 9 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7257
- F1: 0.6761
- Recall: 0.6522
- Precision: 0.7018
- Composite Score: 0.6740 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6988 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 120
- FP: 51
- FN: 64
- TN: 119

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7203, f1=0.6316, recall=0.6792, precision=0.5902, score=0.6731, precisionAware=0.6286  TP:36 FP:25 FN:17 TN:50
- epoch 57: auc=0.7243, f1=0.6226, recall=0.6226, precision=0.6226, score=0.6430, precisionAware=0.6430  TP:33 FP:20 FN:20 TN:55
- epoch 58: auc=0.7205, f1=0.6226, recall=0.6226, precision=0.6226, score=0.6422, precisionAware=0.6422  TP:33 FP:20 FN:20 TN:55
- epoch 56: auc=0.7228, f1=0.6168, recall=0.6226, precision=0.6111, score=0.6409, precisionAware=0.6352  TP:33 FP:21 FN:20 TN:54

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7243, f1=0.6226, recall=0.6226, precision=0.6226, precisionAware=0.6430, composite=0.6430  TP:33 FP:20 FN:20 TN:55
- epoch 58: auc=0.7205, f1=0.6226, recall=0.6226, precision=0.6226, precisionAware=0.6422, composite=0.6422  TP:33 FP:20 FN:20 TN:55
- epoch 56: auc=0.7228, f1=0.6168, recall=0.6226, precision=0.6111, precisionAware=0.6352, composite=0.6409  TP:33 FP:21 FN:20 TN:54
- epoch 59: auc=0.7203, f1=0.6316, recall=0.6792, precision=0.5902, precisionAware=0.6286, composite=0.6731  TP:36 FP:25 FN:17 TN:50
