# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed13 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 13 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7441
- F1: 0.6027
- Recall: 0.6286
- Precision: 0.5789
- Composite Score: 0.6439 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6191 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 66
- FP: 48
- FN: 39
- TN: 103

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 46: auc=0.7187, f1=0.6325, recall=0.6981, precision=0.5781, score=0.6825, precisionAware=0.6226  TP:37 FP:27 FN:16 TN:48
- epoch 28: auc=0.7263, f1=0.6372, recall=0.6792, precision=0.6000, score=0.6760, precisionAware=0.6364  TP:36 FP:24 FN:17 TN:51
- epoch 52: auc=0.7036, f1=0.6000, recall=0.6792, precision=0.5373, score=0.6604, precisionAware=0.5894  TP:36 FP:31 FN:17 TN:44
- epoch 41: auc=0.7316, f1=0.6286, recall=0.6226, precision=0.6346, score=0.6462, precisionAware=0.6522  TP:33 FP:19 FN:20 TN:56

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 27: auc=0.7482, f1=0.5455, recall=0.4528, precision=0.6857, precisionAware=0.6561, composite=0.5397  TP:24 FP:11 FN:29 TN:64
- epoch 22: auc=0.7535, f1=0.5745, recall=0.5094, precision=0.6585, precisionAware=0.6523, composite=0.5777  TP:27 FP:14 FN:26 TN:61
- epoch 41: auc=0.7316, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6522, composite=0.6462  TP:33 FP:19 FN:20 TN:56
- epoch 25: auc=0.7416, f1=0.5918, recall=0.5472, precision=0.6444, precisionAware=0.6481, composite=0.5995  TP:29 FP:16 FN:24 TN:59
