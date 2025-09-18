# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6780
- F1: 0.4331
- Recall: 0.3238
## Confusion matrix (TP/FP/FN/TN)
- TP: 34
- FP: 18
- FN: 71
- TN: 133

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 39: auc=0.7348, f1=0.6182, recall=0.6415, score=0.6532  TP: 34 FP: 23 FN: 19 TN: 52
- epoch 41: auc=0.7399, f1=0.5435, recall=0.4717, score=0.5469  TP: 25 FP: 14 FN: 28 TN: 61
- epoch 35: auc=0.7379, f1=0.5227, recall=0.4340, score=0.5214  TP: 23 FP: 12 FN: 30 TN: 63
- epoch 40: auc=0.7379, f1=0.5227, recall=0.4340, score=0.5214  TP: 23 FP: 12 FN: 30 TN: 63
