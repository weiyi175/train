# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed39 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 39 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7489
- F1: 0.7107
- Recall: 0.7609
- Precision: 0.6667
- Composite Score: 0.7434 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6963 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 140
- FP: 70
- FN: 44
- TN: 100

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.6948, f1=0.6462, recall=0.7925, precision=0.5455, score=0.7290, precisionAware=0.6055  TP:42 FP:35 FN:11 TN:40
- epoch 58: auc=0.6938, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7081, precisionAware=0.6085  TP:40 FP:32 FN:13 TN:43
- epoch 52: auc=0.6762, f1=0.6107, recall=0.7547, precision=0.5128, score=0.6958, precisionAware=0.5749  TP:40 FP:38 FN:13 TN:37
- epoch 60: auc=0.7006, f1=0.6240, recall=0.7358, precision=0.5417, score=0.6953, precisionAware=0.5982  TP:39 FP:33 FN:14 TN:42

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 5: auc=0.5623, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6236, composite=0.1330  TP:1 FP:0 FN:52 TN:75
- epoch 56: auc=0.6911, f1=0.6333, recall=0.7170, precision=0.5672, precisionAware=0.6118, composite=0.6867  TP:38 FP:29 FN:15 TN:46
- epoch 58: auc=0.6938, f1=0.6400, recall=0.7547, precision=0.5556, precisionAware=0.6085, composite=0.7081  TP:40 FP:32 FN:13 TN:43
- epoch 59: auc=0.6948, f1=0.6462, recall=0.7925, precision=0.5455, precisionAware=0.6055, composite=0.7290  TP:42 FP:35 FN:11 TN:40
