# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6633
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 12: auc=0.6390, f1=0.2034, recall=0.1132, score=0.2454  TP: 6 FP: 0 FN: 47 TN: 75
- epoch 11: auc=0.6367, f1=0.1695, recall=0.0943, score=0.2254  TP: 5 FP: 1 FN: 48 TN: 74
- epoch 10: auc=0.6415, f1=0.1379, recall=0.0755, score=0.2074  TP: 4 FP: 1 FN: 49 TN: 74
- epoch 7: auc=0.6473, f1=0.1034, recall=0.0566, score=0.1888  TP: 3 FP: 2 FN: 50 TN: 73
