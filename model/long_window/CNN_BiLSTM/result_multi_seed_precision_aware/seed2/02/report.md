# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed2 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 2 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6917
- F1: 0.6019
- Recall: 0.5217
- Precision: 0.7111
- Composite Score: 0.5798 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6745 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 96
- FP: 39
- FN: 88
- TN: 131

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7127, f1=0.6357, recall=0.7736, precision=0.5395, score=0.7200, precisionAware=0.6030  TP:41 FP:35 FN:12 TN:40
- epoch 25: auc=0.7122, f1=0.6260, recall=0.7736, precision=0.5256, score=0.7170, precisionAware=0.5930  TP:41 FP:37 FN:12 TN:38
- epoch 37: auc=0.7167, f1=0.6667, recall=0.7358, precision=0.6094, score=0.7113, precisionAware=0.6480  TP:39 FP:25 FN:14 TN:50
- epoch 28: auc=0.7318, f1=0.6609, recall=0.7170, precision=0.6129, score=0.7031, precisionAware=0.6511  TP:38 FP:24 FN:15 TN:51

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.7479, f1=0.6604, recall=0.6604, precision=0.6604, precisionAware=0.6779, composite=0.6779  TP:35 FP:18 FN:18 TN:57
- epoch 51: auc=0.7394, f1=0.6538, recall=0.6415, precision=0.6667, precisionAware=0.6774, composite=0.6648  TP:34 FP:17 FN:19 TN:58
- epoch 54: auc=0.7366, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6532, composite=0.6472  TP:33 FP:19 FN:20 TN:56
- epoch 57: auc=0.7384, f1=0.6486, recall=0.6792, precision=0.6207, precisionAware=0.6526, composite=0.6819  TP:36 FP:22 FN:17 TN:53
