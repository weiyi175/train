# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed8 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 8 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7406
- F1: 0.5753
- Recall: 0.4565
- Precision: 0.7778
- Composite Score: 0.5490 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.7096 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 84
- FP: 24
- FN: 100
- TN: 146

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7930, f1=0.6972, recall=0.7170, precision=0.6786, score=0.7263, precisionAware=0.7071  TP:38 FP:18 FN:15 TN:57
- epoch 47: auc=0.7899, f1=0.6607, recall=0.6981, precision=0.6271, score=0.7053, precisionAware=0.6698  TP:37 FP:22 FN:16 TN:53
- epoch 52: auc=0.7879, f1=0.6372, recall=0.6792, precision=0.6000, score=0.6884, precisionAware=0.6487  TP:36 FP:24 FN:17 TN:51
- epoch 27: auc=0.7668, f1=0.6731, recall=0.6604, precision=0.6863, score=0.6855, precisionAware=0.6984  TP:35 FP:16 FN:18 TN:59

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.8035, f1=0.6531, recall=0.6038, precision=0.7111, precisionAware=0.7122, composite=0.6585  TP:32 FP:13 FN:21 TN:62
- epoch 26: auc=0.7592, f1=0.6667, recall=0.6226, precision=0.7174, precisionAware=0.7105, composite=0.6632  TP:33 FP:13 FN:20 TN:62
- epoch 43: auc=0.7930, f1=0.6972, recall=0.7170, precision=0.6786, precisionAware=0.7071, composite=0.7263  TP:38 FP:18 FN:15 TN:57
- epoch 29: auc=0.7764, f1=0.6667, recall=0.6415, precision=0.6939, precisionAware=0.7022, composite=0.6760  TP:34 FP:15 FN:19 TN:60
