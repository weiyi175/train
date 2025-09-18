# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed17 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 17 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7299
- F1: 0.6306
- Recall: 0.6667
- Precision: 0.5983
- Composite Score: 0.6685 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6343 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 70
- FP: 47
- FN: 35
- TN: 104

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7270, f1=0.5983, recall=0.6604, precision=0.5469, score=0.6551, precisionAware=0.5983  TP:35 FP:29 FN:18 TN:46
- epoch 55: auc=0.7170, f1=0.5926, recall=0.6038, precision=0.5818, score=0.6231, precisionAware=0.6121  TP:32 FP:23 FN:21 TN:52
- epoch 52: auc=0.7167, f1=0.5766, recall=0.6038, precision=0.5517, score=0.6182, precisionAware=0.5922  TP:32 FP:26 FN:21 TN:49
- epoch 57: auc=0.7064, f1=0.5714, recall=0.6038, precision=0.5424, score=0.6146, precisionAware=0.5839  TP:32 FP:27 FN:21 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7366, f1=0.6061, recall=0.5660, precision=0.6522, precisionAware=0.6552, composite=0.6122  TP:30 FP:16 FN:23 TN:59
- epoch 46: auc=0.7389, f1=0.5882, recall=0.5660, precision=0.6122, precisionAware=0.6304, composite=0.6073  TP:30 FP:19 FN:23 TN:56
- epoch 26: auc=0.7125, f1=0.5116, recall=0.4151, precision=0.6667, precisionAware=0.6293, composite=0.5035  TP:22 FP:11 FN:31 TN:64
- epoch 23: auc=0.7122, f1=0.5333, recall=0.4528, precision=0.6486, precisionAware=0.6268, composite=0.5289  TP:24 FP:13 FN:29 TN:62
