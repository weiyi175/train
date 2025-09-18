# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7629
- F1: 0.5806
- Recall: 0.5143
## Confusion matrix (TP/FP/FN/TN)
- TP: 54
- FP: 27
- FN: 51
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 46: auc=0.7462, f1=0.6095, recall=0.6038, score=0.6340  TP: 32 FP: 20 FN: 21 TN: 55
- epoch 62: auc=0.7791, f1=0.6139, recall=0.5849, score=0.6324  TP: 31 FP: 17 FN: 22 TN: 58
- epoch 37: auc=0.7218, f1=0.6019, recall=0.5849, score=0.6174  TP: 31 FP: 19 FN: 22 TN: 56
- epoch 58: auc=0.7477, f1=0.6061, recall=0.5660, score=0.6144  TP: 30 FP: 16 FN: 23 TN: 59
