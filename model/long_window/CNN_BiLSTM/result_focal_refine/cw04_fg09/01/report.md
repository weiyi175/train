# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6643
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.6030, f1=0.3797, recall=0.2830, score=0.3760  TP: 15 FP: 11 FN: 38 TN: 64
- epoch 9: auc=0.5982, f1=0.3158, recall=0.2264, score=0.3276  TP: 12 FP: 11 FN: 41 TN: 64
- epoch 11: auc=0.6025, f1=0.3056, recall=0.2075, score=0.3159  TP: 11 FP: 8 FN: 42 TN: 67
- epoch 7: auc=0.5975, f1=0.2727, recall=0.1698, score=0.2862  TP: 9 FP: 4 FN: 44 TN: 71
