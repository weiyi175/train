# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg03 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7388
- F1: 0.4970
- Recall: 0.3905
## Confusion matrix (TP/FP/FN/TN)
- TP: 41
- FP: 19
- FN: 64
- TN: 132

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 47: auc=0.7371, f1=0.5859, recall=0.5472, score=0.5968  TP: 29 FP: 17 FN: 24 TN: 58
- epoch 55: auc=0.7454, f1=0.5455, recall=0.5094, score=0.5674  TP: 27 FP: 19 FN: 26 TN: 56
- epoch 48: auc=0.7311, f1=0.5208, recall=0.4717, score=0.5383  TP: 25 FP: 18 FN: 28 TN: 57
- epoch 46: auc=0.7522, f1=0.5333, recall=0.4528, score=0.5369  TP: 24 FP: 13 FN: 29 TN: 62
