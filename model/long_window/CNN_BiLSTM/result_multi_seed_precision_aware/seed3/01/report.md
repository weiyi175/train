# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed3 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 3 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6757
- F1: 0.5395
- Recall: 0.5524
- Precision: 0.5273
- Composite Score: 0.5732 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5606 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 58
- FP: 52
- FN: 47
- TN: 99

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 51: auc=0.7625, f1=0.6441, recall=0.7170, precision=0.5846, score=0.7042, precisionAware=0.6380  TP:38 FP:27 FN:15 TN:48
- epoch 39: auc=0.7311, f1=0.6549, recall=0.6981, precision=0.6167, score=0.6917, precisionAware=0.6510  TP:37 FP:23 FN:16 TN:52
- epoch 41: auc=0.7348, f1=0.6491, recall=0.6981, precision=0.6066, score=0.6908, precisionAware=0.6450  TP:37 FP:24 FN:16 TN:51
- epoch 58: auc=0.7522, f1=0.6306, recall=0.6604, precision=0.6034, score=0.6698, precisionAware=0.6414  TP:35 FP:23 FN:18 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.7457, f1=0.6022, recall=0.5283, precision=0.7000, precisionAware=0.6798, composite=0.5939  TP:28 FP:12 FN:25 TN:63
- epoch 38: auc=0.7391, f1=0.6105, recall=0.5472, precision=0.6905, precisionAware=0.6762, composite=0.6046  TP:29 FP:13 FN:24 TN:62
- epoch 52: auc=0.7688, f1=0.5957, recall=0.5283, precision=0.6829, precisionAware=0.6739, composite=0.5966  TP:28 FP:13 FN:25 TN:62
- epoch 50: auc=0.7696, f1=0.6200, recall=0.5849, precision=0.6596, precisionAware=0.6697, composite=0.6324  TP:31 FP:16 FN:22 TN:59
