# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg05 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7118
- F1: 0.0721
- Recall: 0.0381
## Confusion matrix (TP/FP/FN/TN)
- TP: 4
- FP: 2
- FN: 101
- TN: 149

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 46: auc=0.6981, f1=0.3562, recall=0.2453, score=0.3691  TP: 13 FP: 7 FN: 40 TN: 68
- epoch 44: auc=0.7074, f1=0.2985, recall=0.1887, score=0.3254  TP: 10 FP: 4 FN: 43 TN: 71
- epoch 45: auc=0.7026, f1=0.2857, recall=0.1887, score=0.3206  TP: 10 FP: 7 FN: 43 TN: 68
- epoch 47: auc=0.7147, f1=0.2687, recall=0.1698, score=0.3084  TP: 9 FP: 5 FN: 44 TN: 70
