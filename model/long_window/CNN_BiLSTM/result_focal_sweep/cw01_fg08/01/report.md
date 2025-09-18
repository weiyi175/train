# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg08 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7132
- F1: 0.5253
- Recall: 0.4952
## Confusion matrix (TP/FP/FN/TN)
- TP: 52
- FP: 41
- FN: 53
- TN: 110

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 61: auc=0.7514, f1=0.6726, recall=0.7170, score=0.7105  TP: 38 FP: 22 FN: 15 TN: 53
- epoch 54: auc=0.7411, f1=0.6667, recall=0.6981, score=0.6973  TP: 37 FP: 21 FN: 16 TN: 54
- epoch 49: auc=0.7426, f1=0.6325, recall=0.6981, score=0.6873  TP: 37 FP: 27 FN: 16 TN: 48
- epoch 59: auc=0.7459, f1=0.6429, recall=0.6792, score=0.6817  TP: 36 FP: 23 FN: 17 TN: 52
