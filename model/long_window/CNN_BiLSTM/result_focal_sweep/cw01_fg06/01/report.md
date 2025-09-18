# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg06 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6858
- F1: 0.2017
- Recall: 0.1143
## Confusion matrix (TP/FP/FN/TN)
- TP: 12
- FP: 2
- FN: 93
- TN: 149

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 49: auc=0.6891, f1=0.3896, recall=0.2830, score=0.3962  TP: 15 FP: 9 FN: 38 TN: 66
- epoch 44: auc=0.7036, f1=0.3684, recall=0.2642, score=0.3833  TP: 14 FP: 9 FN: 39 TN: 66
- epoch 47: auc=0.7089, f1=0.3636, recall=0.2642, score=0.3830  TP: 14 FP: 10 FN: 39 TN: 65
- epoch 45: auc=0.7127, f1=0.3514, recall=0.2453, score=0.3706  TP: 13 FP: 8 FN: 40 TN: 67
