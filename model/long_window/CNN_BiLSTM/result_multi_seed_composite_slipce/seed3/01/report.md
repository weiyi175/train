# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed3 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 3 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7450
- F1: 0.6836
- Recall: 0.6576
- Precision: 0.7118
- Composite Score: 0.6829 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7100 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 121
- FP: 49
- FN: 63
- TN: 121

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 35: auc=0.6855, f1=0.6047, recall=0.7358, precision=0.5132, score=0.6864, precisionAware=0.5751  TP:39 FP:37 FN:14 TN:38
- epoch 40: auc=0.7351, f1=0.6545, recall=0.6792, precision=0.6316, score=0.6830, precisionAware=0.6592  TP:36 FP:21 FN:17 TN:54
- epoch 39: auc=0.6886, f1=0.6167, recall=0.6981, precision=0.5522, score=0.6718, precisionAware=0.5988  TP:37 FP:30 FN:16 TN:45
- epoch 49: auc=0.7130, f1=0.6261, recall=0.6792, precision=0.5806, score=0.6700, precisionAware=0.6207  TP:36 FP:26 FN:17 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.7351, f1=0.6545, recall=0.6792, precision=0.6316, precisionAware=0.6592, composite=0.6830  TP:36 FP:21 FN:17 TN:54
- epoch 34: auc=0.7210, f1=0.6422, recall=0.6604, precision=0.6250, precisionAware=0.6494, composite=0.6671  TP:35 FP:21 FN:18 TN:54
- epoch 33: auc=0.7233, f1=0.5941, recall=0.5660, precision=0.6250, precisionAware=0.6354, composite=0.6059  TP:30 FP:18 FN:23 TN:57
- epoch 31: auc=0.7092, f1=0.6095, recall=0.6038, precision=0.6154, precisionAware=0.6324, composite=0.6266  TP:32 FP:20 FN:21 TN:55
