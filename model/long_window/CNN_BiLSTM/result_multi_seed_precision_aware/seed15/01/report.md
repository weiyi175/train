# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed15 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 15 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7319
- F1: 0.5455
- Recall: 0.5143
- Precision: 0.5806
- Composite Score: 0.5672 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6003 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 54
- FP: 39
- FN: 51
- TN: 112

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7580, f1=0.6271, recall=0.6981, precision=0.5692, score=0.6888, precisionAware=0.6243  TP:37 FP:28 FN:16 TN:47
- epoch 42: auc=0.7389, f1=0.6116, recall=0.6981, precision=0.5441, score=0.6803, precisionAware=0.6033  TP:37 FP:31 FN:16 TN:44
- epoch 26: auc=0.7447, f1=0.6542, recall=0.6604, precision=0.6481, score=0.6754, precisionAware=0.6693  TP:35 FP:19 FN:18 TN:56
- epoch 44: auc=0.7457, f1=0.6207, recall=0.6792, precision=0.5714, score=0.6750, precisionAware=0.6211  TP:36 FP:27 FN:17 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 35: auc=0.7872, f1=0.6316, recall=0.5660, precision=0.7143, precisionAware=0.7041, composite=0.6299  TP:30 FP:12 FN:23 TN:63
- epoch 37: auc=0.7678, f1=0.6237, recall=0.5472, precision=0.7250, precisionAware=0.7032, composite=0.6142  TP:29 FP:11 FN:24 TN:64
- epoch 54: auc=0.7940, f1=0.6465, recall=0.6038, precision=0.6957, precisionAware=0.7006, composite=0.6546  TP:32 FP:14 FN:21 TN:61
- epoch 39: auc=0.7781, f1=0.6087, recall=0.5283, precision=0.7179, precisionAware=0.6972, composite=0.6024  TP:28 FP:11 FN:25 TN:64
