# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 10 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep_smoke/cw02_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6635
- F1: 0.1207
- Recall: 0.0667
## Confusion matrix (TP/FP/FN/TN)
- TP: 7
- FP: 4
- FN: 98
- TN: 147

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 9: auc=0.6136, f1=0.1071, recall=0.0566, score=0.1832  TP: 3 FP: 0 FN: 50 TN: 75
- epoch 10: auc=0.6204, f1=0.0727, recall=0.0377, score=0.1648  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 8: auc=0.6073, f1=0.0727, recall=0.0377, score=0.1621  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 7: auc=0.6063, f1=0.0370, recall=0.0189, score=0.1418  TP: 1 FP: 0 FN: 52 TN: 75
