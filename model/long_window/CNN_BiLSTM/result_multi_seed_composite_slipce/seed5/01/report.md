# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed5 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 5 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7170
- F1: 0.6235
- Recall: 0.5761
- Precision: 0.6795
- Composite Score: 0.6185 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6702 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 106
- FP: 50
- FN: 78
- TN: 120

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 69: auc=0.7434, f1=0.6718, recall=0.8302, precision=0.5641, score=0.7653, precisionAware=0.6323  TP:44 FP:34 FN:9 TN:41
- epoch 65: auc=0.7618, f1=0.6613, recall=0.7736, precision=0.5775, score=0.7375, precisionAware=0.6395  TP:41 FP:30 FN:12 TN:45
- epoch 61: auc=0.7439, f1=0.6557, recall=0.7547, precision=0.5797, score=0.7229, precisionAware=0.6354  TP:40 FP:29 FN:13 TN:46
- epoch 56: auc=0.7610, f1=0.6667, recall=0.7358, precision=0.6094, score=0.7201, precisionAware=0.6569  TP:39 FP:25 FN:14 TN:50

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7610, f1=0.6667, recall=0.7358, precision=0.6094, precisionAware=0.6569, composite=0.7201  TP:39 FP:25 FN:14 TN:50
- epoch 68: auc=0.7613, f1=0.6435, recall=0.6981, precision=0.5968, precisionAware=0.6437, composite=0.6944  TP:37 FP:25 FN:16 TN:50
- epoch 65: auc=0.7618, f1=0.6613, recall=0.7736, precision=0.5775, precisionAware=0.6395, composite=0.7375  TP:41 FP:30 FN:12 TN:45
- epoch 61: auc=0.7439, f1=0.6557, recall=0.7547, precision=0.5797, precisionAware=0.6354, composite=0.7229  TP:40 FP:29 FN:13 TN:46
