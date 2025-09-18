# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm_seed1 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed 1 --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6959
- F1: 0.5764
- Recall: 0.6286
## Confusion matrix (TP/FP/FN/TN)
- TP: 66
- FP: 58
- FN: 39
- TN: 93

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 35: auc=0.6953, f1=0.6614, recall=0.7925, score=0.7337  TP: 42 FP: 32 FN: 11 TN: 43
- epoch 47: auc=0.7107, f1=0.6457, recall=0.7736, score=0.7226  TP: 41 FP: 33 FN: 12 TN: 42
- epoch 30: auc=0.7011, f1=0.6406, recall=0.7736, score=0.7192  TP: 41 FP: 34 FN: 12 TN: 41
- epoch 57: auc=0.7341, f1=0.6609, recall=0.7170, score=0.7036  TP: 38 FP: 24 FN: 15 TN: 51
