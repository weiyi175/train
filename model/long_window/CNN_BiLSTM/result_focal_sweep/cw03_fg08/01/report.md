# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg08 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7348
- F1: 0.5813
- Recall: 0.5619
## Confusion matrix (TP/FP/FN/TN)
- TP: 59
- FP: 39
- FN: 46
- TN: 112

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 40: auc=0.7434, f1=0.6613, recall=0.7736, score=0.7339  TP: 41 FP: 30 FN: 12 TN: 45
- epoch 42: auc=0.7804, f1=0.6786, recall=0.7170, score=0.7181  TP: 38 FP: 21 FN: 15 TN: 54
- epoch 55: auc=0.7620, f1=0.6496, recall=0.7170, score=0.7058  TP: 38 FP: 26 FN: 15 TN: 49
- epoch 34: auc=0.7620, f1=0.6549, recall=0.6981, score=0.6979  TP: 37 FP: 23 FN: 16 TN: 52
