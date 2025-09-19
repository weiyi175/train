# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed16 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 16 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6679
- F1: 0.6916
- Recall: 0.8043
- Precision: 0.6066
- Composite Score: 0.7432 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6443 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 148
- FP: 96
- FN: 36
- TN: 74

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.4523, f1=0.5763, recall=0.9623, precision=0.4113, score=0.7445, precisionAware=0.4690  TP:51 FP:73 FN:2 TN:2
- epoch 60: auc=0.7240, f1=0.6349, recall=0.7547, precision=0.5479, score=0.7126, precisionAware=0.6093  TP:40 FP:33 FN:13 TN:42
- epoch 59: auc=0.7152, f1=0.6349, recall=0.7547, precision=0.5479, score=0.7109, precisionAware=0.6075  TP:40 FP:33 FN:13 TN:42
- epoch 58: auc=0.7145, f1=0.6240, recall=0.7358, precision=0.5417, score=0.6980, precisionAware=0.6009  TP:39 FP:33 FN:14 TN:42

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7240, f1=0.6349, recall=0.7547, precision=0.5479, precisionAware=0.6093, composite=0.7126  TP:40 FP:33 FN:13 TN:42
- epoch 59: auc=0.7152, f1=0.6349, recall=0.7547, precision=0.5479, precisionAware=0.6075, composite=0.7109  TP:40 FP:33 FN:13 TN:42
- epoch 58: auc=0.7145, f1=0.6240, recall=0.7358, precision=0.5417, precisionAware=0.6009, composite=0.6980  TP:39 FP:33 FN:14 TN:42
- epoch 57: auc=0.7122, f1=0.6129, recall=0.7170, precision=0.5352, precisionAware=0.5939, composite=0.6848  TP:38 FP:33 FN:15 TN:42
