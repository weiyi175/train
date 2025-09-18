# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7263
- F1: 0.4331
- Recall: 0.3238
## Confusion matrix (TP/FP/FN/TN)
- TP: 34
- FP: 18
- FN: 71
- TN: 133

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 43: auc=0.7258, f1=0.6609, recall=0.7170, score=0.7019  TP: 38 FP: 24 FN: 15 TN: 51
- epoch 38: auc=0.7369, f1=0.6667, recall=0.6981, score=0.6964  TP: 37 FP: 21 FN: 16 TN: 54
- epoch 45: auc=0.7550, f1=0.6154, recall=0.6038, score=0.6375  TP: 32 FP: 19 FN: 21 TN: 56
- epoch 36: auc=0.7301, f1=0.6038, recall=0.6038, score=0.6290  TP: 32 FP: 21 FN: 21 TN: 54
