# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed2 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 2 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6561
- F1: 0.6991
- Recall: 0.8587
- Precision: 0.5896
- Composite Score: 0.7703 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6357 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 158
- FP: 110
- FN: 26
- TN: 60

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7287, f1=0.7262, recall=0.9683, precision=0.5810, score=0.8477, precisionAware=0.6541  TP:61 FP:44 FN:2 TN:21
- epoch 68: auc=0.7978, f1=0.7755, recall=0.9048, precision=0.6786, score=0.8446, precisionAware=0.7315  TP:57 FP:27 FN:6 TN:38
- epoch 70: auc=0.7512, f1=0.7516, recall=0.9365, precision=0.6277, score=0.8440, precisionAware=0.6895  TP:59 FP:35 FN:4 TN:30
- epoch 65: auc=0.7758, f1=0.7436, recall=0.9206, precision=0.6237, score=0.8386, precisionAware=0.6901  TP:58 FP:35 FN:5 TN:30

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 56: auc=0.7878, f1=0.7080, recall=0.6349, precision=0.8000, precisionAware=0.7699, composite=0.6874  TP:40 FP:10 FN:23 TN:55
- epoch 16: auc=0.7717, f1=0.6538, recall=0.5397, precision=0.8293, precisionAware=0.7651, composite=0.6203  TP:34 FP:7 FN:29 TN:58
- epoch 60: auc=0.7944, f1=0.7597, recall=0.7778, precision=0.7424, precisionAware=0.7580, composite=0.7757  TP:49 FP:17 FN:14 TN:48
- epoch 40: auc=0.7819, f1=0.7419, recall=0.7302, precision=0.7541, precisionAware=0.7560, composite=0.7440  TP:46 FP:15 FN:17 TN:50
