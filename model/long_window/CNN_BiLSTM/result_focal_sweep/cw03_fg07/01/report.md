# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg07 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6279
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.6131, f1=0.3243, recall=0.2264, score=0.3331  TP: 12 FP: 9 FN: 41 TN: 66
- epoch 8: auc=0.6005, f1=0.3099, recall=0.2075, score=0.3168  TP: 11 FP: 7 FN: 42 TN: 68
- epoch 9: auc=0.6058, f1=0.3056, recall=0.2075, score=0.3166  TP: 11 FP: 8 FN: 42 TN: 67
- epoch 7: auc=0.5942, f1=0.2687, recall=0.1698, score=0.2843  TP: 9 FP: 5 FN: 44 TN: 70
