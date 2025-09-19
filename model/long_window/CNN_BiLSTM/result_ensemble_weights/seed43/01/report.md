# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed43 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 43 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6482
- F1: 0.5763
- Recall: 0.5543
- Precision: 0.6000
- Composite Score: 0.5797 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6025 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 102
- FP: 68
- FN: 82
- TN: 102

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 41: auc=0.7258, f1=0.6306, recall=0.6604, precision=0.6034, score=0.6645, precisionAware=0.6361  TP:35 FP:23 FN:18 TN:52
- epoch 42: auc=0.7316, f1=0.6182, recall=0.6415, precision=0.5965, score=0.6525, precisionAware=0.6300  TP:34 FP:23 FN:19 TN:52
- epoch 40: auc=0.7263, f1=0.6055, recall=0.6226, precision=0.5893, score=0.6382, precisionAware=0.6216  TP:33 FP:23 FN:20 TN:52
- epoch 37: auc=0.6999, f1=0.6168, recall=0.6226, precision=0.6111, score=0.6363, precisionAware=0.6306  TP:33 FP:21 FN:20 TN:54

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 45: auc=0.7401, f1=0.5895, recall=0.5283, precision=0.6667, precisionAware=0.6582, composite=0.5890  TP:28 FP:14 FN:25 TN:61
- epoch 49: auc=0.7577, f1=0.6078, recall=0.5849, precision=0.6327, precisionAware=0.6502, composite=0.6264  TP:31 FP:18 FN:22 TN:57
- epoch 48: auc=0.7530, f1=0.6000, recall=0.5660, precision=0.6383, precisionAware=0.6497, composite=0.6136  TP:30 FP:17 FN:23 TN:58
- epoch 58: auc=0.7665, f1=0.5859, recall=0.5472, precision=0.6304, precisionAware=0.6443, composite=0.6027  TP:29 FP:17 FN:24 TN:58
