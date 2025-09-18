# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg06 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6459
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 8: auc=0.5947, f1=0.0000, recall=0.0000, score=0.1189  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 7: auc=0.5902, f1=0.0000, recall=0.0000, score=0.1180  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 17: auc=0.5902, f1=0.0000, recall=0.0000, score=0.1180  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 9: auc=0.5892, f1=0.0000, recall=0.0000, score=0.1178  TP: 0 FP: 0 FN: 53 TN: 75
