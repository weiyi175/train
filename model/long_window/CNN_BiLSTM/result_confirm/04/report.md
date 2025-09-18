# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7285
- F1: 0.6174
- Recall: 0.6762
## Confusion matrix (TP/FP/FN/TN)
- TP: 71
- FP: 54
- FN: 34
- TN: 97

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 52: auc=0.7374, f1=0.6667, recall=0.7170, score=0.7060  TP: 38 FP: 23 FN: 15 TN: 52
- epoch 62: auc=0.7477, f1=0.6496, recall=0.7170, score=0.7029  TP: 38 FP: 26 FN: 15 TN: 49
- epoch 70: auc=0.7442, f1=0.6435, recall=0.6981, score=0.6909  TP: 37 FP: 25 FN: 16 TN: 50
- epoch 54: auc=0.7575, f1=0.6372, recall=0.6792, score=0.6823  TP: 36 FP: 24 FN: 17 TN: 51
