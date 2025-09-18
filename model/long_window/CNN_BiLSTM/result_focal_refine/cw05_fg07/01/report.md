# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6679
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 12: auc=0.6088, f1=0.2769, recall=0.1698, score=0.2897  TP: 9 FP: 3 FN: 44 TN: 72
- epoch 11: auc=0.6108, f1=0.2258, recall=0.1321, score=0.2559  TP: 7 FP: 2 FN: 46 TN: 73
- epoch 10: auc=0.6075, f1=0.1695, recall=0.0943, score=0.2195  TP: 5 FP: 1 FN: 48 TN: 74
- epoch 9: auc=0.6116, f1=0.1404, recall=0.0755, score=0.2022  TP: 4 FP: 0 FN: 49 TN: 75
