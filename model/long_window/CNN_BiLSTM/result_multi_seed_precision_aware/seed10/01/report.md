# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed10 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 10 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6653
- F1: 0.5914
- Recall: 0.7238
- Precision: 0.5000
- Composite Score: 0.6724 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5605 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 76
- FP: 76
- FN: 29
- TN: 75

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.6956, f1=0.6667, recall=0.8679, precision=0.5412, score=0.7731, precisionAware=0.6097  TP:46 FP:39 FN:7 TN:36
- epoch 40: auc=0.7550, f1=0.7009, recall=0.7736, precision=0.6406, score=0.7480, precisionAware=0.6816  TP:41 FP:23 FN:12 TN:52
- epoch 45: auc=0.7260, f1=0.6269, recall=0.7925, precision=0.5185, score=0.7295, precisionAware=0.5925  TP:42 FP:39 FN:11 TN:36
- epoch 56: auc=0.7927, f1=0.6981, recall=0.6981, precision=0.6981, score=0.7170, precisionAware=0.7170  TP:37 FP:16 FN:16 TN:59

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 46: auc=0.7884, f1=0.6598, recall=0.6038, precision=0.7273, precisionAware=0.7193, composite=0.6575  TP:32 FP:12 FN:21 TN:63
- epoch 56: auc=0.7927, f1=0.6981, recall=0.6981, precision=0.6981, precisionAware=0.7170, composite=0.7170  TP:37 FP:16 FN:16 TN:59
- epoch 58: auc=0.7935, f1=0.6857, recall=0.6792, precision=0.6923, precisionAware=0.7106, composite=0.7040  TP:36 FP:16 FN:17 TN:59
- epoch 55: auc=0.7967, f1=0.6600, recall=0.6226, precision=0.7021, precisionAware=0.7084, composite=0.6687  TP:33 FP:14 FN:20 TN:61
