# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed34 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 34 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7092
- F1: 0.6667
- Recall: 0.6902
- Precision: 0.6447
- Composite Score: 0.6869 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6642 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 127
- FP: 70
- FN: 57
- TN: 100

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5404, f1=0.5786, recall=0.8679, precision=0.4340, score=0.7156, precisionAware=0.4986  TP:46 FP:60 FN:7 TN:15
- epoch 59: auc=0.7250, f1=0.6400, recall=0.7547, precision=0.5556, score=0.7144, precisionAware=0.6148  TP:40 FP:32 FN:13 TN:43
- epoch 58: auc=0.7273, f1=0.6349, recall=0.7547, precision=0.5479, score=0.7133, precisionAware=0.6099  TP:40 FP:33 FN:13 TN:42
- epoch 56: auc=0.7449, f1=0.6667, recall=0.7170, precision=0.6230, score=0.7075, precisionAware=0.6605  TP:38 FP:23 FN:15 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7449, f1=0.6667, recall=0.7170, precision=0.6230, precisionAware=0.6605, composite=0.7075  TP:38 FP:23 FN:15 TN:52
- epoch 60: auc=0.7431, f1=0.6549, recall=0.6981, precision=0.6167, precisionAware=0.6534, composite=0.6941  TP:37 FP:23 FN:16 TN:52
- epoch 57: auc=0.7424, f1=0.6549, recall=0.6981, precision=0.6167, precisionAware=0.6533, composite=0.6940  TP:37 FP:23 FN:16 TN:52
- epoch 48: auc=0.7321, f1=0.6609, recall=0.7170, precision=0.6129, precisionAware=0.6511, composite=0.7032  TP:38 FP:24 FN:15 TN:51
