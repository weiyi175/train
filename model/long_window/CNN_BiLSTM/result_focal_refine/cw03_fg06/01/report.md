# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7439
- F1: 0.5980
- Recall: 0.5810
## Confusion matrix (TP/FP/FN/TN)
- TP: 61
- FP: 38
- FN: 44
- TN: 113

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 59: auc=0.8118, f1=0.6847, recall=0.7170, score=0.7263  TP: 38 FP: 20 FN: 15 TN: 55
- epoch 53: auc=0.8156, f1=0.7048, recall=0.6981, score=0.7236  TP: 37 FP: 15 FN: 16 TN: 60
- epoch 49: auc=0.8083, f1=0.6916, recall=0.6981, score=0.7182  TP: 37 FP: 17 FN: 16 TN: 58
- epoch 55: auc=0.8184, f1=0.6931, recall=0.6604, score=0.7018  TP: 35 FP: 13 FN: 18 TN: 62
