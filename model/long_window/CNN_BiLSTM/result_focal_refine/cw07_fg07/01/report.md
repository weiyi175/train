# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.5679
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.5608, f1=0.1724, recall=0.0943, score=0.2110  TP: 5 FP: 0 FN: 48 TN: 75
- epoch 9: auc=0.5638, f1=0.1404, recall=0.0755, score=0.1926  TP: 4 FP: 0 FN: 49 TN: 75
- epoch 8: auc=0.5660, f1=0.1071, recall=0.0566, score=0.1737  TP: 3 FP: 0 FN: 50 TN: 75
- epoch 6: auc=0.5766, f1=0.0714, recall=0.0377, score=0.1556  TP: 2 FP: 1 FN: 51 TN: 74
