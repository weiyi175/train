# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed30 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 30 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7064
- F1: 0.6667
- Recall: 0.6739
- Precision: 0.6596
- Composite Score: 0.6782 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6711 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 124
- FP: 64
- FN: 60
- TN: 106

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 44: auc=0.6797, f1=0.6349, recall=0.7547, precision=0.5479, score=0.7038, precisionAware=0.6004  TP:40 FP:33 FN:13 TN:42
- epoch 46: auc=0.6898, f1=0.6441, recall=0.7170, precision=0.5846, score=0.6897, precisionAware=0.6235  TP:38 FP:27 FN:15 TN:48
- epoch 45: auc=0.6873, f1=0.6333, recall=0.7170, precision=0.5672, score=0.6859, precisionAware=0.6110  TP:38 FP:29 FN:15 TN:46
- epoch 49: auc=0.7016, f1=0.6435, recall=0.6981, precision=0.5968, score=0.6824, precisionAware=0.6318  TP:37 FP:25 FN:16 TN:50

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7192, f1=0.6364, recall=0.6604, precision=0.6140, precisionAware=0.6418, composite=0.6649  TP:35 FP:22 FN:18 TN:53
- epoch 2: auc=0.5872, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6393, composite=0.1581  TP:2 FP:0 FN:51 TN:75
- epoch 52: auc=0.7157, f1=0.6306, recall=0.6604, precision=0.6034, precisionAware=0.6341, composite=0.6625  TP:35 FP:23 FN:18 TN:52
- epoch 60: auc=0.7132, f1=0.6306, recall=0.6604, precision=0.6034, precisionAware=0.6336, composite=0.6620  TP:35 FP:23 FN:18 TN:52
