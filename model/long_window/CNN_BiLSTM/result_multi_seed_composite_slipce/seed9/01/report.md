# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed9 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 9 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7205
- F1: 0.6788
- Recall: 0.7120
- Precision: 0.6485
- Composite Score: 0.7037 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6720 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 131
- FP: 71
- FN: 53
- TN: 99

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 63: auc=0.7047, f1=0.5812, recall=0.6415, precision=0.5312, score=0.6360, precisionAware=0.5809  TP:34 FP:30 FN:19 TN:45
- epoch 70: auc=0.6946, f1=0.5714, recall=0.6415, precision=0.5152, score=0.6311, precisionAware=0.5679  TP:34 FP:32 FN:19 TN:43
- epoch 33: auc=0.6481, f1=0.5556, recall=0.6604, precision=0.4795, score=0.6265, precisionAware=0.5360  TP:35 FP:38 FN:18 TN:37
- epoch 65: auc=0.6848, f1=0.5690, recall=0.6226, precision=0.5238, score=0.6190, precisionAware=0.5696  TP:33 FP:30 FN:20 TN:45

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.6966, f1=0.5510, recall=0.5094, precision=0.6000, precisionAware=0.6046, composite=0.5593  TP:27 FP:18 FN:26 TN:57
- epoch 59: auc=0.7074, f1=0.5545, recall=0.5283, precision=0.5833, precisionAware=0.5995, composite=0.5720  TP:28 FP:20 FN:25 TN:55
- epoch 27: auc=0.6541, f1=0.5510, recall=0.5094, precision=0.6000, precisionAware=0.5961, composite=0.5508  TP:27 FP:18 FN:26 TN:57
- epoch 58: auc=0.7084, f1=0.5660, recall=0.5660, precision=0.5660, precisionAware=0.5945, composite=0.5945  TP:30 FP:23 FN:23 TN:52
