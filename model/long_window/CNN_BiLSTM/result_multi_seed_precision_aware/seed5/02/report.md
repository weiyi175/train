# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed5 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 5 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6696
- F1: 0.6302
- Recall: 0.6576
- Precision: 0.6050
- Composite Score: 0.6518 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6255 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 121
- FP: 79
- FN: 63
- TN: 91

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7467, f1=0.6609, recall=0.7170, precision=0.6129, score=0.7061, precisionAware=0.6540  TP:38 FP:24 FN:15 TN:51
- epoch 38: auc=0.7343, f1=0.6441, recall=0.7170, precision=0.5846, score=0.6986, precisionAware=0.6324  TP:38 FP:27 FN:15 TN:48
- epoch 40: auc=0.7087, f1=0.6179, recall=0.7170, precision=0.5429, score=0.6856, precisionAware=0.5985  TP:38 FP:32 FN:15 TN:43
- epoch 36: auc=0.7472, f1=0.6207, recall=0.6792, precision=0.5714, score=0.6753, precisionAware=0.6214  TP:36 FP:27 FN:17 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7716, f1=0.6122, recall=0.5660, precision=0.6667, precisionAware=0.6713, composite=0.6210  TP:30 FP:15 FN:23 TN:60
- epoch 41: auc=0.7673, f1=0.6415, recall=0.6415, precision=0.6415, precisionAware=0.6667, composite=0.6667  TP:34 FP:19 FN:19 TN:56
- epoch 39: auc=0.7585, f1=0.5895, recall=0.5283, precision=0.6667, precisionAware=0.6619, composite=0.5927  TP:28 FP:14 FN:25 TN:61
- epoch 60: auc=0.7467, f1=0.6609, recall=0.7170, precision=0.6129, precisionAware=0.6540, composite=0.7061  TP:38 FP:24 FN:15 TN:51
