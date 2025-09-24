# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed5 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 5 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6731
- F1: 0.6820
- Recall: 0.8043
- Precision: 0.5920
- Composite Score: 0.7414 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6352 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 148
- FP: 102
- FN: 36
- TN: 68

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7788, f1=0.7273, recall=0.7619, precision=0.6957, score=0.7549, precisionAware=0.7218  TP:48 FP:21 FN:15 TN:44
- epoch 41: auc=0.7751, f1=0.6906, recall=0.7619, precision=0.6316, score=0.7432, precisionAware=0.6780  TP:48 FP:28 FN:15 TN:37
- epoch 50: auc=0.7900, f1=0.7302, recall=0.7302, precision=0.7302, score=0.7421, precisionAware=0.7421  TP:46 FP:17 FN:17 TN:48
- epoch 70: auc=0.7331, f1=0.6957, recall=0.7619, precision=0.6400, score=0.7363, precisionAware=0.6753  TP:48 FP:27 FN:15 TN:38

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 62: auc=0.7905, f1=0.7458, recall=0.6984, precision=0.8000, precisionAware=0.7818, composite=0.7310  TP:44 FP:11 FN:19 TN:54
- epoch 31: auc=0.7560, f1=0.7018, recall=0.6349, precision=0.7843, precisionAware=0.7539, composite=0.6792  TP:40 FP:11 FN:23 TN:54
- epoch 40: auc=0.7680, f1=0.6727, recall=0.5873, precision=0.7872, precisionAware=0.7490, composite=0.6491  TP:37 FP:10 FN:26 TN:55
- epoch 61: auc=0.7726, f1=0.6957, recall=0.6349, precision=0.7692, precisionAware=0.7478, composite=0.6807  TP:40 FP:12 FN:23 TN:53
