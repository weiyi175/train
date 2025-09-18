# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep_60e/run_01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7203
- F1: 0.5027
- Recall: 0.4381
## Confusion matrix (TP/FP/FN/TN)
- TP: 46
- FP: 32
- FN: 59
- TN: 119

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 56: auc=0.7643, f1=0.6667, recall=0.7170, score=0.7113  TP: 38 FP: 23 FN: 15 TN: 52
- epoch 59: auc=0.7260, f1=0.6393, recall=0.7358, score=0.7049  TP: 39 FP: 30 FN: 14 TN: 45
- epoch 57: auc=0.7575, f1=0.6667, recall=0.6981, score=0.7006  TP: 37 FP: 21 FN: 16 TN: 54
- epoch 55: auc=0.7766, f1=0.6667, recall=0.6604, score=0.6855  TP: 35 FP: 17 FN: 18 TN: 58
