# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7396
- F1: 0.5567
- Recall: 0.5143
## Confusion matrix (TP/FP/FN/TN)
- TP: 54
- FP: 35
- FN: 51
- TN: 116

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 56: auc=0.7874, f1=0.5957, recall=0.5283, score=0.6004  TP: 28 FP: 13 FN: 25 TN: 62
- epoch 48: auc=0.7877, f1=0.5745, recall=0.5094, score=0.5846  TP: 27 FP: 14 FN: 26 TN: 61
- epoch 58: auc=0.7877, f1=0.5745, recall=0.5094, score=0.5846  TP: 27 FP: 14 FN: 26 TN: 61
- epoch 41: auc=0.7658, f1=0.5625, recall=0.5094, score=0.5766  TP: 27 FP: 16 FN: 26 TN: 59
