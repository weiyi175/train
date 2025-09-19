# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed2 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 2 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6684
- F1: 0.6932
- Recall: 0.8043
- Precision: 0.6091
- Composite Score: 0.7438 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6462 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 148
- FP: 95
- FN: 36
- TN: 75

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7140, f1=0.6512, recall=0.7925, precision=0.5526, score=0.7344, precisionAware=0.6145  TP:42 FP:34 FN:11 TN:41
- epoch 1: auc=0.4818, f1=0.5731, recall=0.9245, precision=0.4153, score=0.7305, precisionAware=0.4759  TP:49 FP:69 FN:4 TN:6
- epoch 51: auc=0.6943, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7082, precisionAware=0.6086  TP:40 FP:32 FN:13 TN:43
- epoch 59: auc=0.7114, f1=0.6393, recall=0.7358, precision=0.5652, score=0.7020, precisionAware=0.6167  TP:39 FP:30 FN:14 TN:45

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.5673, f1=0.1404, recall=0.0755, precision=1.0000, precisionAware=0.6556, composite=0.1933  TP:4 FP:0 FN:49 TN:75
- epoch 7: auc=0.5726, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6256, composite=0.1351  TP:1 FP:0 FN:52 TN:75
- epoch 6: auc=0.5718, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6255, composite=0.1349  TP:1 FP:0 FN:52 TN:75
- epoch 58: auc=0.7094, f1=0.6261, recall=0.6792, precision=0.5806, precisionAware=0.6200, composite=0.6693  TP:36 FP:26 FN:17 TN:49
