# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7040
- F1: 0.5000
- Recall: 0.4190
## Confusion matrix (TP/FP/FN/TN)
- TP: 44
- FP: 27
- FN: 61
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 53: auc=0.7711, f1=0.6909, recall=0.7170, score=0.7200  TP: 38 FP: 19 FN: 15 TN: 56
- epoch 56: auc=0.7630, f1=0.6549, recall=0.6981, score=0.6981  TP: 37 FP: 23 FN: 16 TN: 52
- epoch 52: auc=0.7753, f1=0.6606, recall=0.6792, score=0.6929  TP: 36 FP: 20 FN: 17 TN: 55
- epoch 60: auc=0.7751, f1=0.6422, recall=0.6604, score=0.6779  TP: 35 FP: 21 FN: 18 TN: 54
