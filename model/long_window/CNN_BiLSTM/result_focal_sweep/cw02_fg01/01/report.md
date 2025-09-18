# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw02_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7296
- F1: 0.6383
- Recall: 0.7143
## Confusion matrix (TP/FP/FN/TN)
- TP: 75
- FP: 55
- FN: 30
- TN: 96

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 61: auc=0.7152, f1=0.6281, recall=0.7170, score=0.6900  TP: 38 FP: 30 FN: 15 TN: 45
- epoch 68: auc=0.7142, f1=0.6281, recall=0.7170, score=0.6898  TP: 38 FP: 30 FN: 15 TN: 45
- epoch 45: auc=0.6808, f1=0.6142, recall=0.7358, score=0.6883  TP: 39 FP: 35 FN: 14 TN: 40
- epoch 60: auc=0.7258, f1=0.6271, recall=0.6981, score=0.6823  TP: 37 FP: 28 FN: 16 TN: 47
