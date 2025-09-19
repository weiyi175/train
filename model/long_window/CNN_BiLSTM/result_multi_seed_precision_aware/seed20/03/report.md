# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed20 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 20 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7217
- F1: 0.6704
- Recall: 0.6467
- Precision: 0.6959
- Composite Score: 0.6688 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6934 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 119
- FP: 52
- FN: 65
- TN: 118

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7628, f1=0.6357, recall=0.7736, precision=0.5395, score=0.7300, precisionAware=0.6130  TP:41 FP:35 FN:12 TN:40
- epoch 59: auc=0.7660, f1=0.6557, recall=0.7547, precision=0.5797, score=0.7273, precisionAware=0.6398  TP:40 FP:29 FN:13 TN:46
- epoch 31: auc=0.7177, f1=0.6260, recall=0.7736, precision=0.5256, score=0.7181, precisionAware=0.5942  TP:41 FP:37 FN:12 TN:38
- epoch 55: auc=0.7708, f1=0.6667, recall=0.7170, precision=0.6230, score=0.7127, precisionAware=0.6656  TP:38 FP:23 FN:15 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7912, f1=0.6538, recall=0.6415, precision=0.6667, precisionAware=0.6877, composite=0.6751  TP:34 FP:17 FN:19 TN:58
- epoch 52: auc=0.7806, f1=0.6122, recall=0.5660, precision=0.6667, precisionAware=0.6731, composite=0.6228  TP:30 FP:15 FN:23 TN:60
- epoch 58: auc=0.7950, f1=0.6346, recall=0.6226, precision=0.6471, precisionAware=0.6729, composite=0.6607  TP:33 FP:18 FN:20 TN:57
- epoch 55: auc=0.7708, f1=0.6667, recall=0.7170, precision=0.6230, precisionAware=0.6656, composite=0.7127  TP:38 FP:23 FN:15 TN:52
