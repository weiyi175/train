# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg04 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7400
- F1: 0.4161
- Recall: 0.2952
## Confusion matrix (TP/FP/FN/TN)
- TP: 31
- FP: 13
- FN: 74
- TN: 138

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 51: auc=0.7369, f1=0.6071, recall=0.6415, score=0.6503  TP: 34 FP: 25 FN: 19 TN: 50
- epoch 49: auc=0.7454, f1=0.6019, recall=0.5849, score=0.6221  TP: 31 FP: 19 FN: 22 TN: 56
- epoch 44: auc=0.7436, f1=0.5918, recall=0.5472, score=0.5999  TP: 29 FP: 16 FN: 24 TN: 59
- epoch 57: auc=0.7333, f1=0.5800, recall=0.5472, score=0.5943  TP: 29 FP: 18 FN: 24 TN: 57
