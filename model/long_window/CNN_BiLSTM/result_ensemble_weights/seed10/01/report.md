# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed10 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 10 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7090
- F1: 0.6649
- Recall: 0.6630
- Precision: 0.6667
- Composite Score: 0.6728 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6746 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 61
- FN: 62
- TN: 109

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 51: auc=0.6928, f1=0.6241, recall=0.8302, precision=0.5000, score=0.7409, precisionAware=0.5758  TP:44 FP:44 FN:9 TN:31
- epoch 46: auc=0.6936, f1=0.6269, recall=0.7925, precision=0.5185, score=0.7230, precisionAware=0.5860  TP:42 FP:39 FN:11 TN:36
- epoch 41: auc=0.6755, f1=0.6269, recall=0.7925, precision=0.5185, score=0.7194, precisionAware=0.5824  TP:42 FP:39 FN:11 TN:36
- epoch 40: auc=0.6765, f1=0.6222, recall=0.7925, precision=0.5122, score=0.7182, precisionAware=0.5781  TP:42 FP:40 FN:11 TN:35

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7210, f1=0.6018, recall=0.6415, precision=0.5667, precisionAware=0.6081, composite=0.6455  TP:34 FP:26 FN:19 TN:49
- epoch 57: auc=0.7031, f1=0.6357, recall=0.7736, precision=0.5395, precisionAware=0.6011, composite=0.7181  TP:41 FP:35 FN:12 TN:40
- epoch 58: auc=0.7140, f1=0.6116, recall=0.6981, precision=0.5441, precisionAware=0.5983, composite=0.6753  TP:37 FP:31 FN:16 TN:44
- epoch 54: auc=0.7137, f1=0.5818, recall=0.6038, precision=0.5614, precisionAware=0.5980, composite=0.6192  TP:32 FP:25 FN:21 TN:50
