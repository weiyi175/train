# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg04 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6589
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 14: auc=0.5955, f1=0.1404, recall=0.0755, score=0.1989  TP: 4 FP: 0 FN: 49 TN: 75
- epoch 13: auc=0.5897, f1=0.1404, recall=0.0755, score=0.1978  TP: 4 FP: 0 FN: 49 TN: 75
- epoch 12: auc=0.5844, f1=0.1404, recall=0.0755, score=0.1967  TP: 4 FP: 0 FN: 49 TN: 75
- epoch 11: auc=0.5857, f1=0.1071, recall=0.0566, score=0.1776  TP: 3 FP: 0 FN: 50 TN: 75
