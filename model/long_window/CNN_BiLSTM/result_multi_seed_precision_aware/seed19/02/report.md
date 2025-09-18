# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed19 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 19 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7171
- F1: 0.6160
- Recall: 0.6952
- Precision: 0.5530
- Composite Score: 0.6759 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6048 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 73
- FP: 59
- FN: 32
- TN: 92

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7374, f1=0.6765, recall=0.8679, precision=0.5542, score=0.7844, precisionAware=0.6275  TP:46 FP:37 FN:7 TN:38
- epoch 35: auc=0.7436, f1=0.6667, recall=0.8491, precision=0.5488, score=0.7733, precisionAware=0.6231  TP:45 FP:37 FN:8 TN:38
- epoch 45: auc=0.7575, f1=0.6720, recall=0.7925, precision=0.5833, score=0.7493, precisionAware=0.6448  TP:42 FP:30 FN:11 TN:45
- epoch 39: auc=0.7603, f1=0.6557, recall=0.7547, precision=0.5797, score=0.7261, precisionAware=0.6386  TP:40 FP:29 FN:13 TN:46

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.7839, f1=0.6604, recall=0.6604, precision=0.6604, precisionAware=0.6851, composite=0.6851  TP:35 FP:18 FN:18 TN:57
- epoch 48: auc=0.7678, f1=0.6481, recall=0.6604, precision=0.6364, precisionAware=0.6662, composite=0.6782  TP:35 FP:20 FN:18 TN:55
- epoch 38: auc=0.7633, f1=0.6545, recall=0.6792, precision=0.6316, precisionAware=0.6648, composite=0.6886  TP:36 FP:21 FN:17 TN:54
- epoch 54: auc=0.7721, f1=0.5979, recall=0.5472, precision=0.6591, precisionAware=0.6633, composite=0.6074  TP:29 FP:15 FN:24 TN:60
