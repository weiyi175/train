# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed21 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 21 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6779
- F1: 0.6588
- Recall: 0.7554
- Precision: 0.5840
- Composite Score: 0.7109 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6252 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 139
- FP: 99
- FN: 45
- TN: 71

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7092, f1=0.6769, recall=0.8302, precision=0.5714, score=0.7600, precisionAware=0.6306  TP:44 FP:33 FN:9 TN:42
- epoch 60: auc=0.7147, f1=0.6720, recall=0.7925, precision=0.5833, score=0.7408, precisionAware=0.6362  TP:42 FP:30 FN:11 TN:45
- epoch 54: auc=0.7107, f1=0.6667, recall=0.7925, precision=0.5753, score=0.7384, precisionAware=0.6298  TP:42 FP:31 FN:11 TN:44
- epoch 58: auc=0.7316, f1=0.6610, recall=0.7358, precision=0.6000, score=0.7125, precisionAware=0.6446  TP:39 FP:26 FN:14 TN:49

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7406, f1=0.6346, recall=0.6226, precision=0.6471, precisionAware=0.6620, composite=0.6498  TP:33 FP:18 FN:20 TN:57
- epoch 52: auc=0.7351, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6529, composite=0.6469  TP:33 FP:19 FN:20 TN:56
- epoch 53: auc=0.7341, f1=0.6486, recall=0.6792, precision=0.6207, precisionAware=0.6518, composite=0.6810  TP:36 FP:22 FN:17 TN:53
- epoch 55: auc=0.7223, f1=0.6549, recall=0.6981, precision=0.6167, precisionAware=0.6492, composite=0.6900  TP:37 FP:23 FN:16 TN:52
