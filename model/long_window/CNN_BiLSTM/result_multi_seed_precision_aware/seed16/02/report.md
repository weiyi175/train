# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed16 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 16 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7394
- F1: 0.7233
- Recall: 0.8098
- Precision: 0.6535
- Composite Score: 0.7698 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6916 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 149
- FP: 79
- FN: 35
- TN: 91

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 38: auc=0.7281, f1=0.6400, recall=0.9057, precision=0.4948, score=0.7904, precisionAware=0.5850  TP:48 FP:49 FN:5 TN:26
- epoch 49: auc=0.7718, f1=0.6870, recall=0.8491, precision=0.5769, score=0.7850, precisionAware=0.6489  TP:45 FP:33 FN:8 TN:42
- epoch 56: auc=0.7781, f1=0.6767, recall=0.8491, precision=0.5625, score=0.7832, precisionAware=0.6399  TP:45 FP:35 FN:8 TN:40
- epoch 44: auc=0.7738, f1=0.6618, recall=0.8491, precision=0.5422, score=0.7778, precisionAware=0.6244  TP:45 FP:38 FN:8 TN:37

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 28: auc=0.7809, f1=0.6667, recall=0.6415, precision=0.6939, precisionAware=0.7031, composite=0.6769  TP:34 FP:15 FN:19 TN:60
- epoch 39: auc=0.7842, f1=0.6792, recall=0.6792, precision=0.6792, precisionAware=0.7002, composite=0.7002  TP:36 FP:17 FN:17 TN:58
- epoch 50: auc=0.7862, f1=0.6602, recall=0.6415, precision=0.6800, precisionAware=0.6953, composite=0.6760  TP:34 FP:16 FN:19 TN:59
- epoch 34: auc=0.7633, f1=0.6847, recall=0.7170, precision=0.6552, precisionAware=0.6856, composite=0.7166  TP:38 FP:20 FN:15 TN:55
