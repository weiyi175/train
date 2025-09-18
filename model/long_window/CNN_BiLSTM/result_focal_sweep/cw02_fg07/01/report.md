# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw02_fg07 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7007
- F1: 0.5628
- Recall: 0.5333
## Confusion matrix (TP/FP/FN/TN)
- TP: 56
- FP: 38
- FN: 49
- TN: 113

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 23: auc=0.7001, f1=0.5981, recall=0.6038, score=0.6214  TP: 32 FP: 22 FN: 21 TN: 53
- epoch 38: auc=0.6928, f1=0.5586, recall=0.5849, score=0.5986  TP: 31 FP: 27 FN: 22 TN: 48
- epoch 35: auc=0.6891, f1=0.5586, recall=0.5849, score=0.5978  TP: 31 FP: 27 FN: 22 TN: 48
- epoch 29: auc=0.7094, f1=0.5714, recall=0.5660, score=0.5963  TP: 30 FP: 22 FN: 23 TN: 53
