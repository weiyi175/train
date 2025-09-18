# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw04_fg07 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.2 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7018
- F1: 0.5251
- Recall: 0.4476
## Confusion matrix (TP/FP/FN/TN)
- TP: 47
- FP: 27
- FN: 58
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 43: auc=0.7442, f1=0.6111, recall=0.6226, score=0.6435  TP: 33 FP: 22 FN: 20 TN: 53
- epoch 41: auc=0.7580, f1=0.6095, recall=0.6038, score=0.6363  TP: 32 FP: 20 FN: 21 TN: 55
- epoch 30: auc=0.7270, f1=0.5849, recall=0.5849, score=0.6133  TP: 31 FP: 22 FN: 22 TN: 53
- epoch 39: auc=0.7565, f1=0.5882, recall=0.5660, score=0.6108  TP: 30 FP: 19 FN: 23 TN: 56
