# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 10 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep_smoke/cw01_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6562
- F1: 0.1681
- Recall: 0.0952
## Confusion matrix (TP/FP/FN/TN)
- TP: 10
- FP: 4
- FN: 95
- TN: 147

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 6: auc=0.6000, f1=0.1053, recall=0.0566, score=0.1799  TP: 3 FP: 1 FN: 50 TN: 74
- epoch 7: auc=0.5937, f1=0.1053, recall=0.0566, score=0.1786  TP: 3 FP: 1 FN: 50 TN: 74
- epoch 8: auc=0.5899, f1=0.1071, recall=0.0566, score=0.1784  TP: 3 FP: 0 FN: 50 TN: 75
- epoch 10: auc=0.5962, f1=0.0727, recall=0.0377, score=0.1599  TP: 2 FP: 0 FN: 51 TN: 75
