# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7352
- F1: 0.5445
- Recall: 0.4952
## Confusion matrix (TP/FP/FN/TN)
- TP: 52
- FP: 34
- FN: 53
- TN: 117

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 52: auc=0.7145, f1=0.5849, recall=0.5849, score=0.6108  TP: 31 FP: 22 FN: 22 TN: 53
- epoch 61: auc=0.7396, f1=0.5714, recall=0.5660, score=0.6024  TP: 30 FP: 22 FN: 23 TN: 53
- epoch 67: auc=0.7411, f1=0.5800, recall=0.5472, score=0.5958  TP: 29 FP: 18 FN: 24 TN: 57
- epoch 54: auc=0.7185, f1=0.5505, recall=0.5660, score=0.5919  TP: 30 FP: 26 FN: 23 TN: 49
