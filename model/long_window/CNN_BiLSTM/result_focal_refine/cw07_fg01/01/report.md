# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7098
- F1: 0.5333
- Recall: 0.4571
## Confusion matrix (TP/FP/FN/TN)
- TP: 48
- FP: 27
- FN: 57
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 62: auc=0.7125, f1=0.6446, recall=0.7358, score=0.7038  TP: 39 FP: 29 FN: 14 TN: 46
- epoch 52: auc=0.7021, f1=0.6281, recall=0.7170, score=0.6873  TP: 38 FP: 30 FN: 15 TN: 45
- epoch 57: auc=0.7356, f1=0.6140, recall=0.6604, score=0.6615  TP: 35 FP: 26 FN: 18 TN: 49
- epoch 49: auc=0.7265, f1=0.6034, recall=0.6604, score=0.6565  TP: 35 FP: 28 FN: 18 TN: 47
