# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce_thresh040/windows_npz.npz --kfold 0 --kfold_seed 42 --final_internal_val_ratio 0.0 --epochs 1 --batch 8 --accumulate_steps 4 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7813
- F1: 0.0274
- Recall: 0.0139
- Precision: 1.0000
- Composite Score: 0.1714 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6645 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 3
- FP: 0
- FN: 213
- TN: 138

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6794, f1=0.2466, recall=0.1429, precision=0.9000, score=0.2813, precisionAware=0.6598  TP:9 FP:1 FN:54 TN:64

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6794, f1=0.2466, recall=0.1429, precision=0.9000, precisionAware=0.6598, composite=0.2813  TP:9 FP:1 FN:54 TN:64
