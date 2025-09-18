# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg08 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7284
- F1: 0.5963
- Recall: 0.6190
## Confusion matrix (TP/FP/FN/TN)
- TP: 65
- FP: 48
- FN: 40
- TN: 103

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 46: auc=0.7462, f1=0.6667, recall=0.7547, score=0.7266  TP: 40 FP: 27 FN: 13 TN: 48
- epoch 52: auc=0.7819, f1=0.6916, recall=0.6981, score=0.7129  TP: 37 FP: 17 FN: 16 TN: 58
- epoch 66: auc=0.7844, f1=0.6789, recall=0.6981, score=0.7096  TP: 37 FP: 19 FN: 16 TN: 56
- epoch 43: auc=0.7213, f1=0.6290, recall=0.7358, score=0.7009  TP: 39 FP: 32 FN: 14 TN: 43
