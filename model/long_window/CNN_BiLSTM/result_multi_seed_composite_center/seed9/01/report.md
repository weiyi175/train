# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed9 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 9 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6959
- F1: 0.6188
- Recall: 0.6087
- Precision: 0.6292
- Composite Score: 0.6292 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6394 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 112
- FP: 66
- FN: 72
- TN: 104

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7953, f1=0.6777, recall=0.7593, precision=0.6119, score=0.7420, precisionAware=0.6683  TP:41 FP:26 FN:13 TN:48
- epoch 69: auc=0.7680, f1=0.6560, recall=0.7593, precision=0.5775, score=0.7300, precisionAware=0.6391  TP:41 FP:30 FN:13 TN:44
- epoch 63: auc=0.7905, f1=0.6609, recall=0.7037, precision=0.6230, score=0.7082, precisionAware=0.6678  TP:38 FP:23 FN:16 TN:51
- epoch 38: auc=0.7898, f1=0.6789, recall=0.6852, precision=0.6727, score=0.7042, precisionAware=0.6980  TP:37 FP:18 FN:17 TN:56

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7840, f1=0.5455, recall=0.3889, precision=0.9130, precisionAware=0.7770, composite=0.5149  TP:21 FP:2 FN:33 TN:72
- epoch 54: auc=0.7880, f1=0.6804, recall=0.6111, precision=0.7674, precisionAware=0.7455, composite=0.6673  TP:33 FP:10 FN:21 TN:64
- epoch 46: auc=0.7883, f1=0.6452, recall=0.5556, precision=0.7692, precisionAware=0.7358, composite=0.6290  TP:30 FP:9 FN:24 TN:65
- epoch 31: auc=0.7743, f1=0.6667, recall=0.5926, precision=0.7619, precisionAware=0.7358, composite=0.6512  TP:32 FP:10 FN:22 TN:64
