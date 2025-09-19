# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed27 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 27 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6509
- F1: 0.6387
- Recall: 0.6630
- Precision: 0.6162
- Composite Score: 0.6533 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6299 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 76
- FN: 62
- TN: 94

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7182, f1=0.6393, recall=0.7358, precision=0.5652, score=0.7034, precisionAware=0.6181  TP:39 FP:30 FN:14 TN:45
- epoch 60: auc=0.7298, f1=0.6491, recall=0.6981, precision=0.6066, score=0.6898, precisionAware=0.6440  TP:37 FP:24 FN:16 TN:51
- epoch 55: auc=0.7203, f1=0.6372, recall=0.6792, precision=0.6000, score=0.6748, precisionAware=0.6352  TP:36 FP:24 FN:17 TN:51
- epoch 54: auc=0.7205, f1=0.6306, recall=0.6604, precision=0.6034, score=0.6635, precisionAware=0.6350  TP:35 FP:23 FN:18 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7338, f1=0.6296, recall=0.6415, precision=0.6182, precisionAware=0.6447, composite=0.6564  TP:34 FP:21 FN:19 TN:54
- epoch 60: auc=0.7298, f1=0.6491, recall=0.6981, precision=0.6066, precisionAware=0.6440, composite=0.6898  TP:37 FP:24 FN:16 TN:51
- epoch 55: auc=0.7203, f1=0.6372, recall=0.6792, precision=0.6000, precisionAware=0.6352, composite=0.6748  TP:36 FP:24 FN:17 TN:51
- epoch 54: auc=0.7205, f1=0.6306, recall=0.6604, precision=0.6034, precisionAware=0.6350, composite=0.6635  TP:35 FP:23 FN:18 TN:52
