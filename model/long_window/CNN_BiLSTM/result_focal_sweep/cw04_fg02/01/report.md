# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw04_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.2 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6597
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 14: auc=0.5995, f1=0.0727, recall=0.0377, score=0.1606  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 13: auc=0.5960, f1=0.0727, recall=0.0377, score=0.1599  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 12: auc=0.5882, f1=0.0370, recall=0.0189, score=0.1382  TP: 1 FP: 0 FN: 52 TN: 75
- epoch 5: auc=0.6272, f1=0.0000, recall=0.0000, score=0.1254  TP: 0 FP: 0 FN: 53 TN: 75
