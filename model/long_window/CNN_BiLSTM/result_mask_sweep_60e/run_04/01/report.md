# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep_60e/run_04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --mask_threshold 0.85 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7345
- F1: 0.5150
- Recall: 0.4095
## Confusion matrix (TP/FP/FN/TN)
- TP: 43
- FP: 19
- FN: 62
- TN: 132

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 42: auc=0.7288, f1=0.6042, recall=0.5472, score=0.6006  TP: 29 FP: 14 FN: 24 TN: 61
- epoch 36: auc=0.7303, f1=0.5918, recall=0.5472, score=0.5972  TP: 29 FP: 16 FN: 24 TN: 59
- epoch 46: auc=0.7356, f1=0.5895, recall=0.5283, score=0.5881  TP: 28 FP: 14 FN: 25 TN: 61
- epoch 48: auc=0.6989, f1=0.5743, recall=0.5472, score=0.5856  TP: 29 FP: 19 FN: 24 TN: 56
