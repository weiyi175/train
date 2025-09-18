# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw04_fg08 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.2 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7128
- F1: 0.5436
- Recall: 0.5048
## Confusion matrix (TP/FP/FN/TN)
- TP: 53
- FP: 37
- FN: 52
- TN: 114

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 58: auc=0.7525, f1=0.6829, recall=0.7925, score=0.7516  TP: 42 FP: 28 FN: 11 TN: 47
- epoch 45: auc=0.7306, f1=0.6400, recall=0.7547, score=0.7155  TP: 40 FP: 32 FN: 13 TN: 43
- epoch 66: auc=0.7452, f1=0.6446, recall=0.7358, score=0.7103  TP: 39 FP: 29 FN: 14 TN: 46
- epoch 67: auc=0.7819, f1=0.6729, recall=0.6792, score=0.6979  TP: 36 FP: 18 FN: 17 TN: 57
