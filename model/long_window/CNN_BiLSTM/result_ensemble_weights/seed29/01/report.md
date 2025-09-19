# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed29 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 29 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7046
- F1: 0.6722
- Recall: 0.6630
- Precision: 0.6816
- Composite Score: 0.6741 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6834 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 57
- FN: 62
- TN: 113

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.3804, f1=0.5549, recall=0.9057, precision=0.4000, score=0.6954, precisionAware=0.4425  TP:48 FP:72 FN:5 TN:3
- epoch 58: auc=0.7039, f1=0.5946, recall=0.6226, precision=0.5690, score=0.6305, precisionAware=0.6036  TP:33 FP:25 FN:20 TN:50
- epoch 57: auc=0.7034, f1=0.5946, recall=0.6226, precision=0.5690, score=0.6304, precisionAware=0.6035  TP:33 FP:25 FN:20 TN:50
- epoch 56: auc=0.6991, f1=0.5714, recall=0.6038, precision=0.5424, score=0.6131, precisionAware=0.5824  TP:32 FP:27 FN:21 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7052, f1=0.5849, recall=0.5849, precision=0.5849, precisionAware=0.6090, composite=0.6090  TP:31 FP:22 FN:22 TN:53
- epoch 58: auc=0.7039, f1=0.5946, recall=0.6226, precision=0.5690, precisionAware=0.6036, composite=0.6305  TP:33 FP:25 FN:20 TN:50
- epoch 57: auc=0.7034, f1=0.5946, recall=0.6226, precision=0.5690, precisionAware=0.6035, composite=0.6304  TP:33 FP:25 FN:20 TN:50
- epoch 59: auc=0.7114, f1=0.5714, recall=0.5660, precision=0.5769, precisionAware=0.6022, composite=0.5967  TP:30 FP:22 FN:23 TN:53
