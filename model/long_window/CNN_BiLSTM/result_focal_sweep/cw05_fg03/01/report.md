# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg03 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7198
- F1: 0.4110
- Recall: 0.2857
## Confusion matrix (TP/FP/FN/TN)
- TP: 30
- FP: 11
- FN: 75
- TN: 140

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 33: auc=0.7054, f1=0.5918, recall=0.5472, score=0.5922  TP: 29 FP: 16 FN: 24 TN: 59
- epoch 40: auc=0.7195, f1=0.5333, recall=0.4528, score=0.5303  TP: 24 FP: 13 FN: 29 TN: 62
- epoch 38: auc=0.7190, f1=0.5217, recall=0.4528, score=0.5267  TP: 24 FP: 15 FN: 29 TN: 60
- epoch 36: auc=0.7301, f1=0.5287, recall=0.4340, score=0.5216  TP: 23 FP: 11 FN: 30 TN: 64
