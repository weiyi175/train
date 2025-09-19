# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --test_windows None --epochs 1 --batch 8 --accumulate_steps 1 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7801
- F1: 0.0729
- Recall: 0.0380
- Precision: 0.8750
- Composite Score: 0.1969 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6154 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 7
- FP: 1
- FN: 177
- TN: 169

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.7801, f1=0.0729, recall=0.0380, precision=0.8750, score=0.1969, precisionAware=0.6154  TP:7 FP:1 FN:177 TN:169

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.7801, f1=0.0729, recall=0.0380, precision=0.8750, precisionAware=0.6154, composite=0.1969  TP:7 FP:1 FN:177 TN:169
