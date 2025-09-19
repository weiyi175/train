# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed32 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 32 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6904
- F1: 0.6833
- Recall: 0.7446
- Precision: 0.6313
- Composite Score: 0.7153 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6587 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 137
- FP: 80
- FN: 47
- TN: 90

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7208, f1=0.6316, recall=0.7925, precision=0.5250, score=0.7299, precisionAware=0.5961  TP:42 FP:38 FN:11 TN:37
- epoch 56: auc=0.7213, f1=0.6508, recall=0.7736, precision=0.5616, score=0.7263, precisionAware=0.6203  TP:41 FP:32 FN:12 TN:43
- epoch 55: auc=0.7273, f1=0.6357, recall=0.7736, precision=0.5395, score=0.7229, precisionAware=0.6059  TP:41 FP:35 FN:12 TN:40
- epoch 49: auc=0.7316, f1=0.6308, recall=0.7736, precision=0.5325, score=0.7223, precisionAware=0.6018  TP:41 FP:36 FN:12 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 48: auc=0.7306, f1=0.6557, recall=0.7547, precision=0.5797, precisionAware=0.6327, composite=0.7202  TP:40 FP:29 FN:13 TN:46
- epoch 53: auc=0.7336, f1=0.6250, recall=0.6604, precision=0.5932, precisionAware=0.6308, composite=0.6644  TP:35 FP:24 FN:18 TN:51
- epoch 60: auc=0.7192, f1=0.6557, recall=0.7547, precision=0.5797, precisionAware=0.6304, composite=0.7179  TP:40 FP:29 FN:13 TN:46
- epoch 54: auc=0.7308, f1=0.6446, recall=0.7358, precision=0.5735, precisionAware=0.6263, composite=0.7075  TP:39 FP:29 FN:14 TN:46
