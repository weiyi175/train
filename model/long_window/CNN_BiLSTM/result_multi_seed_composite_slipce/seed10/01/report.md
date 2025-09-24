# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed10 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 10 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7023
- F1: 0.6387
- Recall: 0.6196
- Precision: 0.6590
- Composite Score: 0.6418 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6615 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 114
- FP: 59
- FN: 70
- TN: 111

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.6800, f1=0.5688, recall=0.5849, precision=0.5536, score=0.5991, precisionAware=0.5834  TP:31 FP:25 FN:22 TN:50
- epoch 34: auc=0.6591, f1=0.5556, recall=0.5660, precision=0.5455, score=0.5815, precisionAware=0.5712  TP:30 FP:25 FN:23 TN:50
- epoch 29: auc=0.6878, f1=0.5524, recall=0.5472, precision=0.5577, score=0.5769, precisionAware=0.5821  TP:29 FP:23 FN:24 TN:52
- epoch 26: auc=0.6810, f1=0.5400, recall=0.5094, precision=0.5745, score=0.5529, precisionAware=0.5854  TP:27 FP:20 FN:26 TN:55

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 28: auc=0.6855, f1=0.4941, recall=0.3962, precision=0.6562, precisionAware=0.6135, composite=0.4835  TP:21 FP:11 FN:32 TN:64
- epoch 24: auc=0.6787, f1=0.5057, recall=0.4151, precision=0.6471, precisionAware=0.6110, composite=0.4950  TP:22 FP:12 FN:31 TN:63
- epoch 39: auc=0.6835, f1=0.5275, recall=0.4528, precision=0.6316, precisionAware=0.6107, composite=0.5214  TP:24 FP:14 FN:29 TN:61
- epoch 32: auc=0.6792, f1=0.5169, recall=0.4340, precision=0.6389, precisionAware=0.6103, composite=0.5079  TP:23 FP:13 FN:30 TN:62
