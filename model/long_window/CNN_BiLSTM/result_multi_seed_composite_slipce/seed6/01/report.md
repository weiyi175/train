# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed6 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 6 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6849
- F1: 0.5613
- Recall: 0.4728
- Precision: 0.6905
- Composite Score: 0.5418 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6506 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 87
- FP: 39
- FN: 97
- TN: 131

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 61: auc=0.7165, f1=0.6087, recall=0.6604, precision=0.5645, score=0.6561, precisionAware=0.6082  TP:35 FP:27 FN:18 TN:48
- epoch 45: auc=0.7064, f1=0.5714, recall=0.6038, precision=0.5424, score=0.6146, precisionAware=0.5839  TP:32 FP:27 FN:21 TN:48
- epoch 69: auc=0.7376, f1=0.5741, recall=0.5849, precision=0.5636, score=0.6122, precisionAware=0.6016  TP:31 FP:24 FN:22 TN:51
- epoch 63: auc=0.7182, f1=0.5794, recall=0.5849, precision=0.5741, score=0.6099, precisionAware=0.6045  TP:31 FP:23 FN:22 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 33: auc=0.7102, f1=0.5591, recall=0.4906, precision=0.6500, precisionAware=0.6348, composite=0.5551  TP:26 FP:14 FN:27 TN:61
- epoch 47: auc=0.7298, f1=0.5882, recall=0.5660, precision=0.6122, precisionAware=0.6286, composite=0.6055  TP:30 FP:19 FN:23 TN:56
- epoch 70: auc=0.7288, f1=0.5275, recall=0.4528, precision=0.6316, precisionAware=0.6198, composite=0.5304  TP:24 FP:14 FN:29 TN:61
- epoch 44: auc=0.7142, f1=0.5657, recall=0.5283, precision=0.6087, precisionAware=0.6169, composite=0.5767  TP:28 FP:18 FN:25 TN:57
