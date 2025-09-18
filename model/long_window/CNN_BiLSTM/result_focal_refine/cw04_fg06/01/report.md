# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7371
- F1: 0.5414
- Recall: 0.4667
## Confusion matrix (TP/FP/FN/TN)
- TP: 49
- FP: 27
- FN: 56
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 49: auc=0.7281, f1=0.6613, recall=0.7736, score=0.7308  TP: 41 FP: 30 FN: 12 TN: 45
- epoch 40: auc=0.7346, f1=0.6838, recall=0.7547, score=0.7294  TP: 40 FP: 24 FN: 13 TN: 51
- epoch 43: auc=0.7333, f1=0.6726, recall=0.7170, score=0.7069  TP: 38 FP: 22 FN: 15 TN: 53
- epoch 50: auc=0.7197, f1=0.6290, recall=0.7358, score=0.7006  TP: 39 FP: 32 FN: 14 TN: 43
