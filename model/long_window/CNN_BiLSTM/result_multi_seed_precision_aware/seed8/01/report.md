# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed8 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 8 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7213
- F1: 0.5973
- Recall: 0.6286
- Precision: 0.5690
- Composite Score: 0.6377 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6079 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 66
- FP: 50
- FN: 39
- TN: 101

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7776, f1=0.7097, recall=0.8302, precision=0.6197, score=0.7835, precisionAware=0.6783  TP:44 FP:27 FN:9 TN:48
- epoch 49: auc=0.7436, f1=0.6667, recall=0.8113, precision=0.5658, score=0.7544, precisionAware=0.6316  TP:43 FP:33 FN:10 TN:42
- epoch 54: auc=0.7414, f1=0.6613, recall=0.7736, precision=0.5775, score=0.7335, precisionAware=0.6354  TP:41 FP:30 FN:12 TN:45
- epoch 52: auc=0.7411, f1=0.6613, recall=0.7736, precision=0.5775, score=0.7334, precisionAware=0.6353  TP:41 FP:30 FN:12 TN:45

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7816, f1=0.6786, recall=0.7170, precision=0.6441, precisionAware=0.6819, composite=0.7184  TP:38 FP:21 FN:15 TN:54
- epoch 59: auc=0.7776, f1=0.7097, recall=0.8302, precision=0.6197, precisionAware=0.6783, composite=0.7835  TP:44 FP:27 FN:9 TN:48
- epoch 51: auc=0.7718, f1=0.6337, recall=0.6038, precision=0.6667, precisionAware=0.6778, composite=0.6464  TP:32 FP:16 FN:21 TN:59
- epoch 39: auc=0.7673, f1=0.6606, recall=0.6792, precision=0.6429, precisionAware=0.6731, composite=0.6912  TP:36 FP:20 FN:17 TN:55
