# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7292
- F1: 0.6276
- Recall: 0.7143
## Confusion matrix (TP/FP/FN/TN)
- TP: 75
- FP: 59
- FN: 30
- TN: 92

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 65: auc=0.7519, f1=0.6271, recall=0.6981, score=0.6876  TP: 37 FP: 28 FN: 16 TN: 47
- epoch 60: auc=0.7409, f1=0.5984, recall=0.7170, score=0.6862  TP: 38 FP: 36 FN: 15 TN: 39
- epoch 58: auc=0.7610, f1=0.6207, recall=0.6792, score=0.6780  TP: 36 FP: 27 FN: 17 TN: 48
- epoch 70: auc=0.7477, f1=0.6207, recall=0.6792, score=0.6754  TP: 36 FP: 27 FN: 17 TN: 48
