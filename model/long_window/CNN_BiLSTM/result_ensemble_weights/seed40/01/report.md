# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed40 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 40 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7062
- F1: 0.7068
- Recall: 0.7663
- Precision: 0.6558
- Composite Score: 0.7364 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6812 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 141
- FP: 74
- FN: 43
- TN: 96

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7421, f1=0.6613, recall=0.7736, precision=0.5775, score=0.7336, precisionAware=0.6355  TP:41 FP:30 FN:12 TN:45
- epoch 59: auc=0.7394, f1=0.6457, recall=0.7736, precision=0.5541, score=0.7284, precisionAware=0.6186  TP:41 FP:33 FN:12 TN:42
- epoch 55: auc=0.7270, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7148, precisionAware=0.6152  TP:40 FP:32 FN:13 TN:43
- epoch 58: auc=0.7394, f1=0.6555, recall=0.7358, precision=0.5909, score=0.7124, precisionAware=0.6400  TP:39 FP:27 FN:14 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7419, f1=0.6491, recall=0.6981, precision=0.6066, precisionAware=0.6464, composite=0.6922  TP:37 FP:24 FN:16 TN:51
- epoch 58: auc=0.7394, f1=0.6555, recall=0.7358, precision=0.5909, precisionAware=0.6400, composite=0.7124  TP:39 FP:27 FN:14 TN:48
- epoch 57: auc=0.7389, f1=0.6306, recall=0.6604, precision=0.6034, precisionAware=0.6387, composite=0.6672  TP:35 FP:23 FN:18 TN:52
- epoch 6: auc=0.6282, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6367, composite=0.1462  TP:1 FP:0 FN:52 TN:75
