# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw04_fg03 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.2 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6971
- F1: 0.2677
- Recall: 0.1619
## Confusion matrix (TP/FP/FN/TN)
- TP: 17
- FP: 5
- FN: 88
- TN: 146

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 35: auc=0.6677, f1=0.4198, recall=0.3208, score=0.4198  TP: 17 FP: 11 FN: 36 TN: 64
- epoch 37: auc=0.6765, f1=0.3896, recall=0.2830, score=0.3937  TP: 15 FP: 9 FN: 38 TN: 66
- epoch 32: auc=0.6926, f1=0.3636, recall=0.2642, score=0.3797  TP: 14 FP: 10 FN: 39 TN: 65
- epoch 33: auc=0.6875, f1=0.3288, recall=0.2264, score=0.3493  TP: 12 FP: 8 FN: 41 TN: 67
