# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed20 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 20 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7036
- F1: 0.6986
- Recall: 0.7935
- Precision: 0.6239
- Composite Score: 0.7470 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6622 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 146
- FP: 88
- FN: 38
- TN: 82

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 48: auc=0.7024, f1=0.6429, recall=0.8491, precision=0.5172, score=0.7579, precisionAware=0.5920  TP:45 FP:42 FN:8 TN:33
- epoch 53: auc=0.7014, f1=0.6338, recall=0.8491, precision=0.5056, score=0.7549, precisionAware=0.5832  TP:45 FP:44 FN:8 TN:31
- epoch 47: auc=0.6916, f1=0.6294, recall=0.8491, precision=0.5000, score=0.7517, precisionAware=0.5771  TP:45 FP:45 FN:8 TN:30
- epoch 52: auc=0.7036, f1=0.6286, recall=0.8302, precision=0.5057, score=0.7444, precisionAware=0.5822  TP:44 FP:43 FN:9 TN:32

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7177, f1=0.6457, recall=0.7736, precision=0.5541, precisionAware=0.6143, composite=0.7240  TP:41 FP:33 FN:12 TN:42
- epoch 57: auc=0.7119, f1=0.6457, recall=0.7736, precision=0.5541, precisionAware=0.6131, composite=0.7229  TP:41 FP:33 FN:12 TN:42
- epoch 56: auc=0.7147, f1=0.6400, recall=0.7547, precision=0.5556, precisionAware=0.6127, composite=0.7123  TP:40 FP:32 FN:13 TN:43
- epoch 55: auc=0.7175, f1=0.6290, recall=0.7358, precision=0.5493, precisionAware=0.6069, composite=0.7001  TP:39 FP:32 FN:14 TN:43
