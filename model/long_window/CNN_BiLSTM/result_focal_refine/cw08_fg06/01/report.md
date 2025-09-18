# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7123
- F1: 0.5871
- Recall: 0.5619
## Confusion matrix (TP/FP/FN/TN)
- TP: 59
- FP: 37
- FN: 46
- TN: 114

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 70: auc=0.7411, f1=0.5859, recall=0.5472, score=0.5976  TP: 29 FP: 17 FN: 24 TN: 58
- epoch 54: auc=0.7281, f1=0.5859, recall=0.5472, score=0.5950  TP: 29 FP: 17 FN: 24 TN: 58
- epoch 58: auc=0.7492, f1=0.5957, recall=0.5283, score=0.5927  TP: 28 FP: 13 FN: 25 TN: 62
- epoch 67: auc=0.7665, f1=0.5684, recall=0.5094, score=0.5786  TP: 27 FP: 15 FN: 26 TN: 60
