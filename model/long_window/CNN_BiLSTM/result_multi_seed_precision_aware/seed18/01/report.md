# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed18 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 18 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7089
- F1: 0.6009
- Recall: 0.6381
- Precision: 0.5678
- Composite Score: 0.6411 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6060 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 67
- FP: 51
- FN: 38
- TN: 100

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7673, f1=0.6667, recall=0.6981, precision=0.6379, score=0.7025, precisionAware=0.6724  TP:37 FP:21 FN:16 TN:54
- epoch 51: auc=0.7401, f1=0.6435, recall=0.6981, precision=0.5968, score=0.6901, precisionAware=0.6395  TP:37 FP:25 FN:16 TN:50
- epoch 54: auc=0.7577, f1=0.6239, recall=0.6415, precision=0.6071, score=0.6595, precisionAware=0.6423  TP:34 FP:22 FN:19 TN:53
- epoch 57: auc=0.7794, f1=0.6286, recall=0.6226, precision=0.6346, score=0.6558, precisionAware=0.6618  TP:33 FP:19 FN:20 TN:56

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7753, f1=0.6186, recall=0.5660, precision=0.6818, precisionAware=0.6815, composite=0.6237  TP:30 FP:14 FN:23 TN:61
- epoch 56: auc=0.7673, f1=0.6667, recall=0.6981, precision=0.6379, precisionAware=0.6724, composite=0.7025  TP:37 FP:21 FN:16 TN:54
- epoch 37: auc=0.7819, f1=0.5618, recall=0.4717, precision=0.6944, precisionAware=0.6721, composite=0.5608  TP:25 FP:11 FN:28 TN:64
- epoch 57: auc=0.7794, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6618, composite=0.6558  TP:33 FP:19 FN:20 TN:56
