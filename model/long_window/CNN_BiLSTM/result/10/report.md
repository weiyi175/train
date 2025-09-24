# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 3 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7739
- F1: 0.0000
- Recall: 0.0000
- Precision: 0.0000
- Composite Score: 0.1548 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.1548 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 184
- TN: 170

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6206, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1241, precisionAware=0.1241  TP:0 FP:0 FN:53 TN:75
- epoch 2: auc=0.6126, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1225, precisionAware=0.1225  TP:0 FP:0 FN:53 TN:75
- epoch 1: auc=0.5909, f1=0.0000, recall=0.0000, precision=0.0000, score=0.1182, precisionAware=0.1182  TP:0 FP:0 FN:53 TN:75

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6206, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1241, composite=0.1241  TP:0 FP:0 FN:53 TN:75
- epoch 2: auc=0.6126, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1225, composite=0.1225  TP:0 FP:0 FN:53 TN:75
- epoch 1: auc=0.5909, f1=0.0000, recall=0.0000, precision=0.0000, precisionAware=0.1182, composite=0.1182  TP:0 FP:0 FN:53 TN:75
